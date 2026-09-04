"""Local-only Gradio application for Push-T data replay and policy rollout."""

from __future__ import annotations

import json
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator

import gradio as gr
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mini_wam.studio.checkpoints import (
    CheckpointValidationError,
    checkpoint_normalization,
    import_checkpoint,
    load_checkpoint,
    scan_checkpoints,
)
from mini_wam.studio.datasets import (
    DatasetSummary,
    DatasetValidationError,
    export_expert_video,
    import_dataset,
)
from mini_wam.studio.overfit import OverfitEvaluator
from mini_wam.studio.pusht import (
    SceneState,
    canvas_to_environment,
    random_scene,
    render_scene,
    run_rollout,
    validate_scene,
)


OUTPUT_ROOT = ROOT / "artifacts" / "studio"


class Runtime:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.overfit_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.running = False
        self.dataset: DatasetSummary = import_dataset("内置数据")
        self.legacy_dataset_fingerprint = self.dataset.fingerprint
        self.overfit_evaluator: OverfitEvaluator | None = None
        models, _ = scan_checkpoints()
        self.selected_model = str(models[0].path) if models else ""


runtime = Runtime()


def _overfit_evaluator() -> OverfitEvaluator:
    with runtime.overfit_lock:
        if runtime.overfit_evaluator is None:
            runtime.overfit_evaluator = OverfitEvaluator(ROOT)
        return runtime.overfit_evaluator


def inspect_overfit_window(slot: int) -> tuple[np.ndarray, np.ndarray, str]:
    if runtime.running:
        raise gr.Error("闭环任务执行中，请结束后再运行过拟合诊断")
    try:
        result = _overfit_evaluator().inspect(int(slot))
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    text = (
        f"### 训练窗口 {result.slot}/64\n"
        f"- checkpoint 中的数据索引：`{result.dataset_index}`\n"
        f"- 来源：episode（回合）{result.episode_id}，第 {result.episode_step} 步；"
        f"全局帧 {result.global_index}\n"
        f"- 有效预测步：{result.valid_steps}/16\n"
        f"- normalized Smooth L1 loss（归一化平滑 L1 损失）：{result.normalized_loss:.6f}\n"
        f"- 16 步平均动作端点误差：{result.action_error_mean:.2f} px（像素）\n"
        f"- 实际执行前 4 步平均动作端点误差：{result.first_four_error_mean:.2f} px\n\n"
        "绿色是专家动作，红色是模型预测。这里使用的就是训练时的两帧输入。"
    )
    return result.history_image, result.action_overlay, text


def evaluate_all_overfit_windows() -> str:
    if runtime.running:
        raise gr.Error("闭环任务执行中，请结束后再运行过拟合诊断")
    result = _overfit_evaluator().evaluate_all()
    reduction = 1.0 - result.stored_final_loss / result.stored_initial_loss
    return (
        "### 64 个训练窗口的重新评估\n"
        f"- checkpoint 保存的初始/最终 loss（损失）："
        f"{result.stored_initial_loss:.6f} → {result.stored_final_loss:.6f}\n"
        f"- 损失下降：{reduction:.2%}\n"
        f"- 当前重新计算 loss：{result.recomputed_loss:.6f}\n"
        f"- 动作端点误差：平均 {result.action_error_mean:.2f} px；"
        f"中位数 {result.action_error_median:.2f} px；P90（第90百分位） {result.action_error_p90:.2f} px\n"
        f"- 前 4 步平均误差：{result.first_four_error_mean:.2f} px\n\n"
        "**判断：模型确实明显拟合了这 64 个训练窗口，不属于完全没学会；"
        "但平均十几像素的残差仍可能在闭环里迅速累积。**"
    )


def execute_overfit_window(
    slot: int,
) -> Generator[tuple[np.ndarray, str, str | None], None, None]:
    with runtime.lock:
        if runtime.running:
            raise gr.Error("已有任务正在执行")
        runtime.running = True
        runtime.stop_event.clear()
    try:
        evaluator = _overfit_evaluator()
        recovered = evaluator.recover_rollout_start(int(slot))
        model, _ = load_checkpoint(evaluator.checkpoint_path)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        video_path = OUTPUT_ROOT / "overfit_rollouts" / f"window-{int(slot):02d}-{run_id}.mp4"
        last_update = None
        for update in run_rollout(
            model,
            evaluator.stats,
            recovered.scene,
            video_path,
            stop_event=runtime.stop_event,
            initial_history=recovered.initial_history,
            scene_is_body_pose=True,
            frame_label="APPROX. TRAINING-STATE RECOVERY",
        ):
            last_update = update
            final_status = "任务未完成" if update.video_path else "执行中"
            metrics = (
                f"> 训练窗口 {int(slot)}；T 块图像恢复 IoU（交并比）："
                f"{recovered.block_mask_iou:.3f}。这是近似恢复，不是原始模拟器状态。\n\n"
                + _metrics_text(update.metrics, final_status)
            )
            yield update.frame, metrics, str(update.video_path) if update.video_path else None
        if last_update is not None:
            metadata = {
                "kind": "approximate_overfit_training_window_rollout",
                "window_slot": int(slot),
                "dataset_index": evaluator.indices[int(slot) - 1],
                "block_mask_iou": recovered.block_mask_iou,
                "recovered_scene_body_pose": _scene_list(recovered.scene),
                "metrics": last_update.metrics.to_dict(),
                "video": str(video_path),
            }
            video_path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except (ValueError, CheckpointValidationError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        with runtime.lock:
            runtime.running = False
            runtime.stop_event.clear()


def _scene_list(scene: SceneState) -> list[float]:
    return scene.as_array().astype(float).tolist()


def _scene(value: list[float]) -> SceneState:
    return SceneState.from_array(np.asarray(value, dtype=np.float32))


def _scene_text(scene: SceneState) -> str:
    return (
        f"智能体：({scene.agent_x:.1f}, {scene.agent_y:.1f})　"
        f"T 块：({scene.block_x:.1f}, {scene.block_y:.1f})　"
        f"角度：{np.degrees(scene.block_angle):.1f}°"
    )


def _dataset_text(summary: DatasetSummary) -> str:
    return (
        "### 数据检查通过\n"
        f"- 数据：`{summary.repo_id}`\n"
        f"- episode（回合）：{summary.episode_count}；帧：{summary.frame_count}；帧率：{summary.fps}\n"
        f"- 图像：{summary.image_shape}；state（状态）：{summary.state_dim} 维；action（动作）：{summary.action_dim} 维\n"
        f"- 动作范围：x {summary.action_min[0]:.1f}–{summary.action_max[0]:.1f}，"
        f"y {summary.action_min[1]:.1f}–{summary.action_max[1]:.1f}\n"
        f"- 位置均值/标准差：{summary.position_mean} / {summary.position_std}\n"
        f"- 动作均值/标准差：{summary.action_mean} / {summary.action_std}\n"
        f"- 数据集指纹：`{summary.fingerprint[:16]}…`\n\n"
        "> 导入数据只用于回放、兼容性检查与归一化，不会训练模型。"
    )


def _model_choices() -> tuple[list[tuple[str, str]], list[str]]:
    models, errors = scan_checkpoints()
    choices = [(f"{item.name} · {item.status}", str(item.path)) for item in models]
    return choices, errors


def refresh_models(current: str) -> tuple[dict, str]:
    choices, errors = _model_choices()
    values = [value for _, value in choices]
    selected = current if current in values else (values[0] if values else "")
    suffix = "" if not errors else "\n\n未载入：" + "；".join(errors)
    return gr.update(choices=choices, value=selected), model_details(selected) + suffix


def model_details(path: str) -> str:
    if not path:
        return "没有找到兼容的 ActionOnlyPolicy checkpoint（检查点）。"
    try:
        _, info = load_checkpoint(path)
    except CheckpointValidationError as exc:
        return f"模型不可用：{exc}"
    legacy_note = "旧格式适配：归一化取自当前数据集" if info.legacy else "新格式 bundle（打包文件）"
    return (
        f"**{info.name}**　状态：{info.status}  \n"
        f"架构：`{info.architecture}`；step（训练步）：{info.step if info.step is not None else '未知'}；"
        f"validation loss（验证损失）：{info.validation_loss if info.validation_loss is not None else '未知'}  \n"
        f"{legacy_note}"
    )


def select_model(path: str, previous: str) -> tuple[str, str, dict]:
    with runtime.lock:
        if runtime.running:
            raise gr.Error("任务执行中不能切换模型；请先停止并等待视频保存。")
        load_checkpoint(path)
        runtime.selected_model = path
    return path, model_details(path), gr.update(value=None)


def upload_model(path: str | None) -> tuple[dict, str, str]:
    if not path:
        raise gr.Error("请先选择 .pt 文件")
    with runtime.lock:
        if runtime.running:
            raise gr.Error("任务执行中不能导入模型")
        try:
            info = import_checkpoint(path)
        except CheckpointValidationError as exc:
            raise gr.Error(str(exc)) from exc
        runtime.selected_model = str(info.path)
    choices, _ = _model_choices()
    return gr.update(choices=choices, value=str(info.path)), model_details(str(info.path)), str(info.path)


def load_data(source_kind: str, source: str) -> tuple[str, dict, dict]:
    with runtime.lock:
        if runtime.running:
            raise gr.Error("任务执行中不能更换数据集")
        try:
            summary = import_dataset(source_kind, source)
        except DatasetValidationError as exc:
            raise gr.Error(str(exc)) from exc
        runtime.dataset = summary
    return (
        _dataset_text(summary),
        summary.to_dict(),
        gr.update(minimum=0, maximum=max(summary.episode_ids), value=0),
    )


def make_expert_video(episode_id: int) -> str:
    summary = runtime.dataset
    output = OUTPUT_ROOT / "expert_replays" / f"{summary.repo_id.replace('/', '__')}-ep{int(episode_id):04d}.mp4"
    try:
        return str(export_expert_video(summary, int(episode_id), output))
    except DatasetValidationError as exc:
        raise gr.Error(str(exc)) from exc


def seed_scene(seed: int) -> tuple[np.ndarray, list[float], str, float]:
    scene = random_scene(int(seed))
    return render_scene(scene), _scene_list(scene), _scene_text(scene), float(np.degrees(scene.block_angle))


def edit_scene(
    current: list[float], target: str, angle_degrees: float, evt: gr.SelectData
) -> tuple[np.ndarray, list[float], str]:
    scene = _scene(current)
    x, y = canvas_to_environment(float(evt.index[0]), float(evt.index[1]))
    values = _scene_list(scene)
    if target == "智能体":
        values[0:2] = [x, y]
    else:
        values[2:5] = [x, y, float(np.radians(angle_degrees))]
    candidate = _scene(values)
    try:
        validate_scene(candidate)
        frame = render_scene(candidate)
    except ValueError as exc:
        raise gr.Error(f"布局无效：{exc}") from exc
    return frame, values, _scene_text(candidate)


def change_angle(current: list[float], angle_degrees: float) -> tuple[np.ndarray, list[float], str]:
    values = list(current)
    values[4] = float(np.radians(angle_degrees))
    scene = _scene(values)
    try:
        validate_scene(scene)
        frame = render_scene(scene)
    except ValueError as exc:
        raise gr.Error(f"布局无效：{exc}") from exc
    return frame, values, _scene_text(scene)


def _metrics_text(metrics, status: str = "执行中") -> str:
    if metrics.is_success:
        status = "任务完成"
    elif metrics.stopped:
        status = "已停止"
    return (
        f"### {status}\n"
        f"- success（成功）：{metrics.is_success}\n"
        f"- coverage（覆盖率）：最终 {metrics.final_coverage:.4f}；最高 {metrics.max_coverage:.4f}\n"
        f"- reward（奖励）累计：{metrics.reward:.4f}\n"
        f"- 环境步数：{metrics.steps}；模型推理次数：{metrics.model_calls}\n"
        f"- 平均推理延迟：{metrics.mean_inference_ms:.2f} ms（毫秒）"
    )


def execute_task(scene_values: list[float], selected_model: str) -> Generator[tuple[np.ndarray, str, str | None], None, None]:
    with runtime.lock:
        if runtime.running:
            raise gr.Error("已有任务正在执行")
        runtime.running = True
        runtime.stop_event.clear()
        dataset = runtime.dataset
        runtime.selected_model = selected_model
    try:
        if not selected_model:
            raise gr.Error("请先选择模型")
        scene = _scene(scene_values)
        validate_scene(scene)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model, _ = load_checkpoint(selected_model, device=device)
        normalization = checkpoint_normalization(
            selected_model,
            dataset.normalization(),
            dataset.fingerprint,
            runtime.legacy_dataset_fingerprint,
        )
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        video_path = OUTPUT_ROOT / "rollouts" / f"{run_id}.mp4"
        last_update = None
        for update in run_rollout(
            model,
            normalization,
            scene,
            video_path,
            stop_event=runtime.stop_event,
        ):
            last_update = update
            final_status = "任务未完成" if update.video_path else "执行中"
            yield (
                update.frame,
                _metrics_text(update.metrics, final_status),
                str(update.video_path) if update.video_path else None,
            )
        if last_update is not None:
            metadata = {
                "model": selected_model,
                "dataset": dataset.to_dict(),
                "scene": _scene_list(scene),
                "metrics": last_update.metrics.to_dict(),
                "video": str(video_path),
            }
            video_path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except (ValueError, CheckpointValidationError) as exc:
        raise gr.Error(str(exc)) from exc
    finally:
        with runtime.lock:
            runtime.running = False
            runtime.stop_event.clear()


def stop_task() -> str:
    runtime.stop_event.set()
    return "正在停止；会先保存目前已经生成的画面。"


def build_app() -> gr.Blocks:
    choices, scan_errors = _model_choices()
    initial_model = runtime.selected_model
    initial_scene = random_scene(0)
    initial_frame = render_scene(initial_scene)
    with gr.Blocks(title="Push-T 模型执行工作台", fill_width=True) as demo:
        gr.Markdown(
            "# Push-T 模型执行工作台\n"
            "在同一个本地页面回放专家数据、切换 ActionOnlyPolicy 模型，并观察模型真实闭环执行。"
        )
        gr.Markdown(
            "> 这里不训练模型。绿色目标固定；只有环境返回 success（成功）时，页面才显示“任务完成”。"
        )
        selected_model_state = gr.State(initial_model)
        scene_state = gr.State(_scene_list(initial_scene))

        with gr.Tab("执行任务"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_dropdown = gr.Dropdown(
                        choices=choices,
                        value=initial_model or None,
                        label="执行模型",
                        info="只能选择当前 ActionOnlyPolicy 架构",
                    )
                    model_status = gr.Markdown(
                        model_details(initial_model)
                        + (("\n\n未载入：" + "；".join(scan_errors)) if scan_errors else "")
                    )
                    with gr.Row():
                        seed = gr.Number(value=0, precision=0, label="seed（随机种子）")
                        random_button = gr.Button("生成随机场景")
                    placement_target = gr.Radio(
                        ["智能体", "T 块"], value="智能体", label="点击画布时移动"
                    )
                    angle = gr.Slider(-180, 180, value=float(np.degrees(initial_scene.block_angle)), label="T 块角度（度）")
                    scene_status = gr.Markdown(_scene_text(initial_scene))
                    with gr.Row():
                        run_button = gr.Button("运行模型", variant="primary")
                        stop_button = gr.Button("停止")
                with gr.Column(scale=3):
                    canvas = gr.Image(
                        initial_frame,
                        label="实时仿真（点击可布置场景）",
                        type="numpy",
                        height=520,
                        interactive=True,
                        buttons=["fullscreen"],
                    )
            with gr.Row():
                rollout_metrics = gr.Markdown("尚未运行。")
                rollout_video = gr.Video(label="闭环执行视频")

        with gr.Tab("过拟合诊断"):
            gr.Markdown(
                "这里直接调出 `action_only_overfit.pt` 保存的 64 个训练窗口。"
                "它检验模型是否记住训练输入，不把专家轨迹回放冒充模型闭环执行。"
            )
            with gr.Row():
                overfit_slot = gr.Slider(
                    1, 64, value=1, step=1, label="训练窗口编号"
                )
                inspect_overfit_button = gr.Button("检查这个训练窗口", variant="primary")
                evaluate_all_button = gr.Button("重新评估全部 64 个")
                run_overfit_button = gr.Button("从这个训练窗口开始做任务")
                stop_overfit_button = gr.Button("停止")
            with gr.Row():
                overfit_history = gr.Image(
                    label="训练输入：左边 t-1，右边 t",
                    type="numpy",
                    height=360,
                    buttons=["fullscreen"],
                )
                overfit_overlay = gr.Image(
                    label="当前帧上的16步动作：专家绿色 / 模型红色",
                    type="numpy",
                    height=360,
                    buttons=["fullscreen"],
                )
            overfit_result = gr.Markdown(
                "选择窗口后点击检查；第一次载入模型和数据会稍慢。\n\n"
                "> 这些样本是专家轨迹中途的窗口。原数据没有保存 T 块内部位姿，"
                "因此不能无损恢复成完全相同的仿真闭环起点。"
            )
            overfit_rollout_status = gr.Markdown(
                "点击“从这个训练窗口开始做任务”后，模型会使用真实的两帧训练输入；"
                "T 块位姿由图像近似恢复。"
            )
            overfit_rollout_video = gr.Video(label="过拟合模型做任务的视频（近似恢复起点）")

        with gr.Tab("专家数据"):
            gr.Markdown("> 专家回放是数据中的历史图像，不是当前模型在仿真中执行。")
            with gr.Row():
                source_kind = gr.Radio(
                    ["内置数据", "Hugging Face 仓库", "本地目录"],
                    value="内置数据",
                    label="数据来源",
                )
                source_value = gr.Textbox(
                    value="lerobot/pusht_image",
                    label="repo ID（仓库标识）或本地目录",
                )
                import_data_button = gr.Button("导入并检查", variant="primary")
            dataset_status = gr.Markdown(_dataset_text(runtime.dataset))
            dataset_json = gr.JSON(runtime.dataset.to_dict(), label="完整数据元信息")
            with gr.Row():
                episode = gr.Slider(
                    0,
                    max(runtime.dataset.episode_ids),
                    value=0,
                    step=1,
                    label="episode（回合）",
                )
                replay_button = gr.Button("生成专家回放")
            expert_video = gr.Video(label="专家回放视频")

        with gr.Tab("模型管理"):
            gr.Markdown(
                "只读取 tensor（张量）权重；任意 pickled Python 模型会被拒绝。"
            )
            upload = gr.File(label="上传 .pt checkpoint（检查点）", file_types=[".pt"], type="filepath")
            with gr.Row():
                upload_button = gr.Button("检查并导入", variant="primary")
                refresh_button = gr.Button("重新扫描项目模型")
            upload_result = gr.Markdown("当前模型目录会自动扫描。")

        random_button.click(
            seed_scene,
            inputs=seed,
            outputs=[canvas, scene_state, scene_status, angle],
        )
        canvas.select(
            edit_scene,
            inputs=[scene_state, placement_target, angle],
            outputs=[canvas, scene_state, scene_status],
        )
        angle.release(
            change_angle,
            inputs=[scene_state, angle],
            outputs=[canvas, scene_state, scene_status],
        )
        model_dropdown.change(
            select_model,
            inputs=[model_dropdown, selected_model_state],
            outputs=[selected_model_state, model_status, rollout_video],
        )
        run_button.click(
            execute_task,
            inputs=[scene_state, selected_model_state],
            outputs=[canvas, rollout_metrics, rollout_video],
            concurrency_limit=1,
        )
        stop_button.click(stop_task, outputs=rollout_metrics, queue=False)
        inspect_overfit_button.click(
            inspect_overfit_window,
            inputs=overfit_slot,
            outputs=[overfit_history, overfit_overlay, overfit_result],
            concurrency_limit=1,
        )
        evaluate_all_button.click(
            evaluate_all_overfit_windows,
            outputs=overfit_result,
            concurrency_limit=1,
        )
        run_overfit_button.click(
            execute_overfit_window,
            inputs=overfit_slot,
            outputs=[overfit_overlay, overfit_rollout_status, overfit_rollout_video],
            concurrency_limit=1,
        )
        stop_overfit_button.click(stop_task, outputs=overfit_rollout_status, queue=False)
        import_data_button.click(
            load_data,
            inputs=[source_kind, source_value],
            outputs=[dataset_status, dataset_json, episode],
        )
        replay_button.click(make_expert_video, inputs=episode, outputs=expert_video)
        upload_button.click(
            upload_model,
            inputs=upload,
            outputs=[model_dropdown, upload_result, selected_model_state],
        )
        refresh_button.click(
            refresh_models,
            inputs=selected_model_state,
            outputs=[model_dropdown, upload_result],
        )
    return demo


demo = build_app()


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo.queue(default_concurrency_limit=2).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        max_file_size="1gb",
        allowed_paths=[str(OUTPUT_ROOT), str(runtime.dataset.root)],
        show_error=True,
    )

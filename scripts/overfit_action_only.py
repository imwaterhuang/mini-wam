"""在固定 64 个训练样本上过拟合 ActionOnlyPolicy。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# 让脚本找到 src/mini_wam，并把缓存留在项目目录中。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"),
)

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split
from mini_wam.models.action_only import ActionOnlyPolicy, masked_smooth_l1_loss


NUM_SAMPLES = 64
BATCH_SIZE = 16
TRAIN_STEPS = 200
LEARNING_RATE = 3e-4
SEED = 0
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "action_only_overfit.pt"


@torch.inference_mode()
def evaluate(model, loader, device):
    """计算固定 64 个样本上的平均 loss。"""
    model.eval()
    loss_sum = 0.0
    valid_step_sum = 0

    for batch in loader:
        pixels = batch["observation_history"].to(device)
        positions = batch["agent_position"].to(device)
        targets = batch["action_chunk"].to(device)
        mask = batch["action_valid_mask"].to(device)

        predictions = model(pixels, positions)
        loss = masked_smooth_l1_loss(predictions, targets, mask)

        valid_steps = mask.sum().item()
        loss_sum += loss.item() * valid_steps
        valid_step_sum += valid_steps

    return loss_sum / valid_step_sum


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 1. 建立完整训练数据集。
    stats = NormalizationStats.from_audit_file(
        PROJECT_ROOT / "artifacts" / "data_audit.json"
    )
    train_episode_ids = load_episode_split(
        PROJECT_ROOT / "splits" / "episodes_seed42.json",
        "train",
    )
    dataset = MiniWAMDataset(
        dataset_root=PROJECT_ROOT / "data" / "lerobot" / "pusht_image",
        episode_ids=train_episode_ids,
        normalization=stats,
    )

    # 2. 固定抽取 64 个样本，并提前读入内存。
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:NUM_SAMPLES].tolist()
    samples = []
    for index in indices:
        sample = dataset[index]
        samples.append(
            {
                "observation_history": sample["observation_history"],
                "agent_position": sample["agent_position"],
                "action_chunk": sample["action_chunk"],
                "action_valid_mask": sample["action_valid_mask"],
            }
        )

    train_loader = DataLoader(samples, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(samples, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 创建模型和优化器。过拟合检查暂时不下载预训练权重。
    model = ActionOnlyPolicy(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    initial_loss = evaluate(model, eval_loader, device)
    print(f"device: {device}")
    print(f"initial loss: {initial_loss:.6f}")

    # 4. 反复训练同一组 64 个样本。
    step = 0
    while step < TRAIN_STEPS:
        for batch in train_loader:
            model.train()

            pixels = batch["observation_history"].to(device)
            positions = batch["agent_position"].to(device)
            targets = batch["action_chunk"].to(device)
            mask = batch["action_valid_mask"].to(device)

            optimizer.zero_grad()
            predictions = model(pixels, positions)
            loss = masked_smooth_l1_loss(predictions, targets, mask)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 25 == 0:
                fixed_loss = evaluate(model, eval_loader, device)
                print(f"step {step:3d}: loss = {fixed_loss:.6f}")

            if step == TRAIN_STEPS:
                break

    final_loss = evaluate(model, eval_loader, device)
    reduction = 1.0 - final_loss / initial_loss

    # 5. 保存模型和优化器。
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "indices": indices,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
        CHECKPOINT_PATH,
    )

    # 6. 重新加载，并确认同一个输入得到相同输出。
    reference_batch = next(iter(eval_loader))
    reference_pixels = reference_batch["observation_history"].to(device)
    reference_positions = reference_batch["agent_position"].to(device)

    model.eval()
    with torch.inference_mode():
        output_before_reload = model(reference_pixels, reference_positions)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    restored_model = ActionOnlyPolicy(pretrained=False).to(device)
    restored_model.load_state_dict(checkpoint["model"])
    restored_model.eval()
    with torch.inference_mode():
        output_after_reload = restored_model(reference_pixels, reference_positions)

    reload_matches = torch.allclose(
        output_before_reload,
        output_after_reload,
        rtol=1e-5,
        atol=1e-6,
    )
    outputs_change_with_input = output_before_reload.flatten(1).std(dim=0).mean().item() > 1e-6

    print(f"final loss: {final_loss:.6f}")
    print(f"loss reduction: {reduction:.1%}")
    print(f"reload matches: {reload_matches}")
    print(f"outputs change with input: {outputs_change_with_input}")

    if reduction < 0.90:
        raise RuntimeError("loss 没有下降 90%")
    if not reload_matches:
        raise RuntimeError("checkpoint 重新加载后的输出不一致")
    if not outputs_change_with_input:
        raise RuntimeError("模型对不同输入输出了近似相同的动作")


if __name__ == "__main__":
    main()

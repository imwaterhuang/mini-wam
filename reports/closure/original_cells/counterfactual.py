import threading, traceback, json, time, torch
from torch.utils.data import DataLoader
from mini_wam.data import MiniWAMDataset, load_episode_split
from mini_wam.models.future_head import masked_cosine_loss

counterfactual_out=run/"finalization"/"counterfactual"
counterfactual_out.mkdir(parents=True,exist_ok=True)

def run_counterfactual():
    start=time.perf_counter()
    progress_file=counterfactual_out/"progress.json"
    try:
        validation_ids=load_episode_split(repo/"splits"/"episodes_seed42.json","validation")
        dataset=MiniWAMDataset(
            repo/"data"/"lerobot"/"pusht_image",
            validation_ids,
            normalization=norm,
            action_horizon=16,
            future_horizon=4,
            include_future_observations=True,
        )
        loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,drop_last=True)
        totals={"correct":0.0,"shuffled":0.0,"reversed":0.0}
        valid_steps=0
        model.eval()
        with torch.inference_mode():
            for batch_index,batch in enumerate(loader):
                obs=batch["observation_history"].to(device)
                pos=batch["agent_position"].to(device)
                prefix=batch["action_chunk"][:,:4].to(device)
                future=batch["future_observations"].to(device)
                mask=batch["future_valid_mask"].to(device)
                visual=model.visual_encoder(obs)
                position_features=model.state_encoder(pos)
                fused=model.history_fusion(visual,position_features)
                target=model.frozen_target_encoder(future)
                shuffled=prefix.roll(1,0)
                raw_prefix=norm.denormalize_action(prefix.cpu())
                raw_position=norm.denormalize_position(pos[:,-1].cpu()).unsqueeze(1)
                reversed_raw=(2.0*raw_position-raw_prefix).clamp(0.0,512.0)
                reversed_prefix=norm.normalize_action(reversed_raw).to(device)
                variants={"correct":prefix,"shuffled":shuffled,"reversed":reversed_prefix}
                count=int(mask.sum().item())
                for name,actions in variants.items():
                    encoded=model.action_prefix_encoder(actions)
                    pred=model.future_head(torch.cat([fused,encoded],dim=-1))
                    totals[name]+=float(masked_cosine_loss(pred,target,mask).item())*count
                valid_steps+=count
                if batch_index%10==0:
                    progress_file.write_text(json.dumps({"status":"running","batches":batch_index+1,"valid_steps":valid_steps}),encoding="utf-8")
        losses={k:v/valid_steps for k,v in totals.items()}
        result={
            "status":"complete",
            "checkpoint_step":int(ck["step"]),
            "split":"offline_validation",
            "windows_evaluated":len(loader)*64,
            "valid_future_steps":valid_steps,
            "conditions":{
                "correct":"matched history and ground-truth action prefix",
                "shuffled":"cyclically shifted action prefix with no self-pair in each full batch",
                "reversed":"absolute action displacement reflected around current agent position, then clipped",
            },
            "mean_masked_cosine_loss":losses,
            "deltas_vs_correct":{k:losses[k]-losses["correct"] for k in ("shuffled","reversed")},
            "elapsed_seconds":time.perf_counter()-start,
            "sealed_final_test_used":False,
        }
        atomic_json(counterfactual_out/"results.json",result)
        progress_file.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print("COUNTERFACTUAL_COMPLETE",json.dumps(result),flush=True)
    except Exception as exc:
        progress_file.write_text(json.dumps({"status":"failed","error":repr(exc),"traceback":traceback.format_exc()},ensure_ascii=False,indent=2),encoding="utf-8")
        print("COUNTERFACTUAL_FAILED",repr(exc),flush=True)

counterfactual_thread=threading.Thread(target=run_counterfactual,name="mini_wam_counterfactual")
counterfactual_thread.start()
print("COUNTERFACTUAL_STARTED",counterfactual_thread.is_alive())

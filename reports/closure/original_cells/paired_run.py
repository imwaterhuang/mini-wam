import csv, gc, hashlib, json, math, shutil, threading, time, traceback
from mini_wam.evaluation.policy import summarize_policy_episodes

compare_out=drive/"comparisons"/"web_random_20260910_20scenes_v1"
(compare_out/"action_only"/"videos").mkdir(parents=True,exist_ok=True)
(compare_out/"future_aware"/"videos").mkdir(parents=True,exist_ok=True)
(compare_out/"successful_videos"/"action_only").mkdir(parents=True,exist_ok=True)
(compare_out/"successful_videos"/"future_aware").mkdir(parents=True,exist_ok=True)

def atomic_json(path,payload):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)

scene_manifest={
    "source":"Push-T web workbench seed_scene using mini_wam.studio.pusht.random_scene",
    "requested_seed_range":[2026091000,2026091021],
    "rejected_invalid_seeds":[2026091000,2026091005],
    "selection_policy":"first 20 valid scenes in ascending seed order; fixed before either model ran",
    "max_steps":300,
    "execute_steps":4,
    "scenes":valid_records,
}
atomic_json(compare_out/"scenes.json",scene_manifest)

def run_paired_comparison():
    started=time.perf_counter()
    results={"action_only":[],"future_aware":[]}
    try:
        for index,record in enumerate(valid_records):
            order=[("action_only",action_model,action_norm),("future_aware",future_model,future_norm)]
            if index%2:
                order.reverse()
            for name,current_model,current_norm in order:
                scene=SceneState(**record["state"])
                video_path=compare_out/name/"videos"/f"{record['scene_id']}.mp4"
                t0=time.perf_counter()
                updates=list(run_rollout(current_model,current_norm,scene,video_path,max_steps=300,execute_steps=4))
                metrics=updates[-1].metrics.to_dict()
                episode={"scene_id":record["scene_id"],"seed":record["seed"],**metrics,"video":str(video_path.relative_to(compare_out)),"elapsed_seconds":time.perf_counter()-t0}
                results[name].append(episode)
                if metrics["is_success"]:
                    shutil.copyfile(video_path,compare_out/"successful_videos"/name/video_path.name)
                del updates
                gc.collect()
            atomic_json(compare_out/"partial_results.json",results)
            progress={
                "status":"running","completed_scene_pairs":index+1,"total_scene_pairs":len(valid_records),
                "action_only_successes":sum(int(e["is_success"]) for e in results["action_only"]),
                "future_aware_successes":sum(int(e["is_success"]) for e in results["future_aware"]),
                "latest_scene":record["scene_id"],"elapsed_seconds":time.perf_counter()-started,
            }
            atomic_json(compare_out/"progress.json",progress)
            print(f"[{index+1}/20] action={progress['action_only_successes']} future={progress['future_aware_successes']}",flush=True)

        by_a={e["scene_id"]:e for e in results["action_only"]}
        by_f={e["scene_id"]:e for e in results["future_aware"]}
        paired={"both_success":0,"action_only_only":0,"future_aware_only":0,"neither_success":0}
        for scene_id in by_a:
            a=by_a[scene_id]["is_success"]; f=by_f[scene_id]["is_success"]
            key="both_success" if a and f else "action_only_only" if a else "future_aware_only" if f else "neither_success"
            paired[key]+=1
        discordant=paired["action_only_only"]+paired["future_aware_only"]
        minority=min(paired["action_only_only"],paired["future_aware_only"])
        p_value=min(1.0,2*sum(math.comb(discordant,k) for k in range(minority+1))/(2**discordant)) if discordant else 1.0
        summary_a=summarize_policy_episodes(results["action_only"])
        summary_f=summarize_policy_episodes(results["future_aware"])
        final={
            "status":"complete",
            "scene_manifest":"scenes.json",
            "protocol":{"source":"web workbench random scenes","episode_count":20,"same_initial_states":True,"same_normalization":same_norm,"max_steps":300,"action_horizon":16,"execute_steps":4,"sealed_final_test_used":False},
            "checkpoints":{
                "action_only":{"step":int(action_ck["step"]),"sha256":hashlib.sha256((action_run/"checkpoints"/"best.pt").read_bytes()).hexdigest()},
                "future_aware":{"step":int(future_ck["step"]),"sha256":hashlib.sha256((future_run/"checkpoints"/"best.pt").read_bytes()).hexdigest()},
            },
            "summary":{"action_only":summary_a,"future_aware":summary_f},
            "success_rate_difference_future_minus_action":summary_f["success_rate"]-summary_a["success_rate"],
            "paired_outcomes":paired,
            "mcnemar_exact_two_sided_p":p_value,
            "episodes":results,
            "elapsed_seconds":time.perf_counter()-started,
        }
        atomic_json(compare_out/"results.json",final)
        for name in ("action_only","future_aware"):
            rows=results[name]
            with (compare_out/name/"episodes.csv").open("w",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        atomic_json(compare_out/"progress.json",{"status":"complete","completed_scene_pairs":20,"total_scene_pairs":20,"action_only_successes":summary_a["success_count"],"future_aware_successes":summary_f["success_count"],"elapsed_seconds":final["elapsed_seconds"]})
        print("PAIRED_COMPARISON_COMPLETE",json.dumps(final["summary"]),flush=True)
    except Exception as exc:
        atomic_json(compare_out/"progress.json",{"status":"failed","error":repr(exc),"traceback":traceback.format_exc()})
        print("PAIRED_COMPARISON_FAILED",repr(exc),flush=True)

comparison_thread=threading.Thread(target=run_paired_comparison,name="web_random_paired_comparison")
comparison_thread.start()
print("COMPARISON_STARTED",comparison_thread.is_alive(),compare_out)

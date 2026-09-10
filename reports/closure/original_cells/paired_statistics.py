import csv, json, math, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

p=Path("/content/drive/MyDrive/mini-wam-runs/comparisons/web_random_20260910_20scenes_v1")
r=json.loads((p/"results.json").read_text())
a={e["scene_id"]:e for e in r["episodes"]["action_only"]}
f={e["scene_id"]:e for e in r["episodes"]["future_aware"]}
rows=[]
for scene in r["episodes"]["action_only"]:
    sid=scene["scene_id"]
    rows.append({
        "scene_id":sid,"seed":scene["seed"],
        "action_only_success":a[sid]["is_success"],"future_aware_success":f[sid]["is_success"],
        "action_only_final_coverage":a[sid]["final_coverage"],"future_aware_final_coverage":f[sid]["final_coverage"],
        "action_only_max_coverage":a[sid]["max_coverage"],"future_aware_max_coverage":f[sid]["max_coverage"],
    })
with (p/"paired_comparison.csv").open("w",newline="",encoding="utf-8") as h:
    w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def wilson(k,n,z=1.959963984540054):
    ph=k/n; d=1+z*z/n
    c=(ph+z*z/(2*n))/d
    m=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return [c-m,c+m]
extra={
    "action_only_success_rate_wilson_95":wilson(2,20),
    "future_aware_success_rate_wilson_95":wilson(5,20),
    "action_only_final_coverage_std":float(np.std([e["final_coverage"] for e in a.values()])),
    "future_aware_final_coverage_std":float(np.std([e["final_coverage"] for e in f.values()])),
}
r["uncertainty_and_dispersion"]=extra
(p/"results.json").write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

fig,axes=plt.subplots(1,2,figsize=(10,4.2))
axes[0].bar(["action-only","future-aware"],[0.10,0.25],color=["#3b82f6","#f97316"])
axes[0].set_ylim(0,1); axes[0].set_ylabel("strict success rate"); axes[0].set_title("20 web-random scenes")
axes[1].boxplot([[e["final_coverage"] for e in a.values()],[e["final_coverage"] for e in f.values()]],tick_labels=["action-only","future-aware"])
axes[1].axhline(.95,color="#16a34a",ls="--",lw=1,label="success threshold")
axes[1].set_ylim(0,1.05); axes[1].set_ylabel("final coverage"); axes[1].set_title("Final coverage distribution"); axes[1].legend()
for ax in axes: ax.grid(axis="y",alpha=.2)
fig.tight_layout(); fig.savefig(p/"comparison.png",dpi=180); plt.close(fig)

all_videos=list((p/"action_only"/"videos").glob("*.mp4"))+list((p/"future_aware"/"videos").glob("*.mp4"))
print("EXTRA",json.dumps(extra))
print("PAIRED_CSV", (p/"paired_comparison.csv").stat().st_size)
print("PLOT", (p/"comparison.png").stat().st_size)
print("VIDEOS",len(all_videos),"ZERO_BYTE",sum(x.stat().st_size==0 for x in all_videos))

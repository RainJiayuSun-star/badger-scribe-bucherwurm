import json,glob,os
os.chdir(os.path.expanduser("~/badger-scribe-bucherwurm/runs"))
for r in ["dec-base-kade103","dec-rp10-kade103","dec-rp10-survey011","dec-rp10-group5","dec-2mp-kade103","dec-2mp-group5","dec-4mp-survey011"]:
    d=json.load(open(f"finetune/{r}/summary.json"))
    print(r,round(d["cer"],4),"loops",d["loop_cuts"],"pages",d["pages"],"min",round(d["wall_seconds"]/60,1),"gib",d["peak_gpu_gib"],d["git_version"][:8])
for f in sorted(glob.glob("analysis/step2/*.json")+glob.glob("analysis/step3/*.json")):
    d=json.load(open(f))
    if "gain" in d: print(f,round(d["base_cer"],4),round(d["arm_cer"],4),round(d["gain"],4),round(d["ci95_low"],4),round(d["ci95_high"],4),d["pages_better"],d["pages_worse"])
s=json.load(open("finetune/submission-02/summary.json"))
print("sub02",s["pages"],len(s["empty_pages"]),s["loop_cuts"],round(s["wall_seconds"]/60,1),s["peak_gpu_gib"],s["git_version"],s["model_id"],[v.split("/")[-2] for v in s["adapters"].values()])
c=json.load(open("finetune/submission-02/config.json")); print(c.get("repetition_penalty"),c["max_pixels"],c["max_pixels_by_category"],c["loop_guard"],c["eval_max_new_tokens"])
for g in [1,6,4,2,3,5]:
    p=f"finetune/ft5-english-dominy_group{g}/status.json"
    if os.path.exists(p):
        d=json.load(open(p)); print("group",g,d.get("phase"),d.get("step"),d.get("epoch"),d.get("started"),d.get("updated"))

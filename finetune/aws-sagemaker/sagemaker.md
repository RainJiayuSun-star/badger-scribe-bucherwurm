# AWS SageMaker workshop: takeaways and agent operating guide

Written 2026-11-25 and updated 2026-11-26 for Ian Gloege. Source: [Machine Learning on AWS with SageMaker and Bedrock](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/index.html).

This is a reusable guide, not a certificate of completion. Episodes 1–7 were completed in this chat. Episode 8's first tuning trial was verified Completed, with validation accuracy 0.7413 and about $0.005 compute at the lesson reference rate. The larger tuning searches are not recorded as launched. Episode 9's data, estimates, and tagged estimator were prepared without launching its GPU job. Episode 11 (Bedrock RAG) stopped partway on a Bedrock permission issue. Episode 12's RAG Processing job was run and scored on 2026-11-26 (section 10), and episode 15's cleanup checks were run the same night (section 12). Episode 13 (GPU notebook) was not run. The fine-tuning job in section 9 was not executed. Recheck current AWS state before resuming.

## 1. The central architecture

Use a small CPU notebook as a controller. Keep durable inputs, checkpoints, and outputs in S3. Submit expensive training and batch inference to separate managed jobs with explicit time limits. These jobs continue independently of the notebook and release their compute when they terminate. Closing a browser, interrupting a cell, restarting a kernel, or stopping the controller does not stop a submitted job.

Choose compute for the workload, not for the notebook. A GPU is useful when a model and batch provide enough work to keep it busy. A tiny model can finish sooner and cost less on a CPU because provisioning, container download, data transfer, and CUDA initialization dominate.

For research, compare campus resources with cloud costs before renting a large GPU. In this UW workshop, stay within the lesson's permitted resources. Do not create endpoints, EC2 fleets, new roles, or unrelated infrastructure to work around a failed exercise.

## 2. This workshop's environment

| Setting | Value |
|---|---|
| AWS region | `us-east-2` |
| S3 bucket | `badgerscribe-gloegeian` |
| Notebook | `BadgerScribe-GloegeIan` |
| Notebook instance / volume | `ml.t3.medium` / 20 GB |
| Execution role | `ml-sagemaker-use` |
| Lesson clone | `/home/ec2-user/SageMaker/ML_with_AWS_SageMaker` |
| Titanic working directory | `/home/ec2-user/SageMaker/titanic` |
| Supplemental fork | `https://github.com/iangloege/AWS_helpers.git` |
| Supplemental clone | `/home/ec2-user/SageMaker/AWS_helpers` |

Required tags: `Name=GloegeIan`, `Project=BadgerScribe`, and a resource-appropriate `Purpose`: `Storage`, `Notebook`, `Training`, `Tuning`, or `Inference`. Jobs need their own tags; do not assume notebook tags propagate. Set tags on the tuner as well as its estimator and verify child-job attribution.

Use the existing region, bucket, and role. A suitable controller setup is:

```python
import boto3
import sagemaker

bucket_name = "badgerscribe-gloegeian"
region = boto3.Session().region_name
role = sagemaker.get_execution_role()
assert region == "us-east-2"
assert role.endswith("/ml-sagemaker-use")
session = sagemaker.Session(default_bucket=bucket_name)
sm = session.sagemaker_client
```

The explicit default bucket keeps SDK staging in the existing bucket rather than silently choosing another one. Use the role's temporary credentials. Never request or embed AWS access keys.

## 3. Storage, reproducibility, and credentials

- S3 is object storage: a prefix resembles a directory but is part of an object's key. Keep raw data, processed inputs, checkpoints, and outputs distinguishable.
- Keep buckets private and use existing IAM permissions. An access error is a reason to inspect the role, policy, path, and region, not to enable public access.
- Work outside the lesson repository for downloaded data and generated artifacts. Keep code changes easy to review.
- Files under `/home/ec2-user/SageMaker` are the notebook's persistent project storage. Kernel variables disappear after a restart; system-environment package changes may need to be repeated after stop/start or maintenance. Do not confuse notebook volume persistence with a training container's temporary filesystem.
- Version code, dependency pins, preprocessing configuration, model revision, random seeds, data split, and job configuration. Record the actual image URI and framework versions used by each run.
- A notebook's installed packages do not define the managed job's environment. Use the lesson's container and its supported dependency mechanism.
- Ignore data, checkpoints, model weights, notebook credentials, and other large/generated files in Git. Review notebook outputs before committing.
- Fine-grained GitHub tokens are suitable when restricted to the intended repository and necessary permissions. A push requires repository Contents write permission. Enter a token only through a hidden prompt such as `getpass`; never put it in code, a remote URL, logs, screenshots, or chat.
- Commit identity (name and email) is separate from authentication.

## 4. Test the script before renting compute

Run a short local test first. Check CLI argument parsing, channel paths, metric printing, dependencies, serialization, and output paths. A local success is evidence about the code, not a reliable prediction of remote runtime.

In a training container, channel inputs appear under `/opt/ml/input/data/<channel>/`. Save final artifacts under `/opt/ml/model`; SageMaker packages that directory into the job's model artifact. Use the artifact URI reported by `describe_training_job`, rather than guessing a path from a job name.

Keep preprocessing identical across training, validation, and test inference. Persist fitted transforms and pass them back when processing new data. Fit learned transforms on the training split only in a production pipeline. The workshop notebook fits its scaler before the train/validation split, which leaks validation information; preserve that distinction when interpreting workshop scores. Never refit the scaler on the final test set.

## 5. Estimate, approve, submit once, and monitor

Before every launch, prepare a reviewable configuration containing:

1. Exact instance type, instances per job, total jobs, and maximum concurrency.
2. Data/model inputs, entry point, container versions, output prefix, and required tags.
3. A measured or calculated runtime estimate and a hard runtime cap.
4. Estimated normal compute cost and the cost at the cap, with rate, region, and exclusions stated.
5. A unique job name, checkpoint location where appropriate, and a monitoring/recovery plan.

Get approval immediately before each metered training, tuning, fine-tuning, processing, GPU-notebook, or Bedrock job. Routine preparation and read-only verification do not need it.

Use the lesson's `helpers.check_launch` and fine-tuning checks before submission. Current account policies and quotas may differ from old lesson examples. Do not change permissions or select an unapproved instance merely to bypass an explicit deny.

Persist the job name immediately after submission. Before retrying an uncertain launch, inspect AWS for the intended job. An explicit unique name can prevent duplicate submissions. With `wait=False`, returning from the notebook means submitted, not completed. Monitor until a terminal status and inspect failure reasons and logs when unsuccessful. Do not silently retry failed metered jobs.

## 6. Cost accounting

For a homogeneous training job:

```text
compute cost ≈ BillableTimeInSeconds × InstanceCount / 3600 × hourly rate
wall time = TrainingEndTime − CreationTime
cap estimate ≈ runtime cap seconds × InstanceCount / 3600 × hourly rate
```

Treat the cap calculation as a compute planning bound, not an invoice guarantee: AWS defines which phases a stopping condition covers, and storage, requests, transfers, and other services are additional. Check the current job API and pricing before launch. Cost Explorer is delayed; the completed job's billable-time record provides an immediate compute estimate.

For small models, time a short run, scale by examples and epochs, then add startup and upload overhead. For large dense-model estimates, the lesson uses parameter count × tokens × passes, approximately 6 FLOPs per parameter-token for full training and 4 for LoRA, adjusted for checkpointing and achieved GPU throughput. This is a rough planning model; vision tokens, activations, sequence length, hardware precision, and attention can change the result substantially. Calibrate against a small real run.

Prices below are lesson reference rates, not verified current regional prices: `ml.m5.large` $0.115/hour, `ml.g4dn.xlarge` $0.74/hour, `ml.g5.xlarge` $1.41/hour. Reprice actual work in `us-east-2`.

## 7. What our CPU, GPU, and distributed runs demonstrated

Same Titanic network, data, 1,000 epochs, and a 1,200-second cap:

| Configuration | Wall minutes | Billable instance-seconds | Compute estimate | Final validation accuracy |
|---|---:|---:|---:|---:|
| 1 × `ml.m5.large` | 3.40 | 164 | $0.00524 | 0.7762 |
| 1 × `ml.g4dn.xlarge` | about 5.0 | about 228 | about $0.046 | 0.7622 |
| 2 × `ml.m5.large`, DDP | 4.08 | 408 total | $0.01303 | 0.8182 |

The GPU billable time is rounded from the displayed 3.8 minutes; do not treat 228 seconds as an exact API measurement. The scores are single runs with different initialization behavior, not proof that hardware changes model quality.

CPU job: `BadgerScribe-GloegeIan-nn-2026-11-25-09-12-10-807`.

GPU job: `BadgerScribe-GloegeIan-nn-2026-11-25-10-31-48-821`.

DDP job: `BadgerScribe-GloegeIan-nn-ddp-2026-11-25-11-24-40-608`.

All three were verified Completed. GPU startup made it slower and about nine times as expensive as the CPU baseline. DDP also cost more and took longer for this tiny network.

`instance_count=2` alone does not distribute an ordinary custom training script. This lesson's DDP script joins a process group, shards rows, and wraps the model in DistributedDataParallel. The MPI launcher starts one process per CPU host. Logs verified rank 0 using 285 of 569 rows and rank 1 using 284, with rank 0 saving the model. Gradient synchronization has a cost; more workers pay only when enough computation is saved. Try a suitable single device before multi-device or multi-host training.

## 8. Hyperparameter tuning

A tuner combines an estimator, parameter ranges, an objective metric, and a trial budget. The script must accept every tuned argument and print a metric whose regex and name match the tuner. Start with one trial before a larger search. Here, the lesson searches epochs `{100, 1000, 10000}` and learning rate `0.001–0.1`, maximizing `validation:accuracy`.

`max_jobs` is the total number of trials. `max_parallel_jobs` controls concurrency; `instance_count` controls instances inside each child job. Sum child-job compute costs. Parallel trials finish sooner but give Bayesian selection less completed evidence to use. Do not assume startup or instance reuse savings without verifying the actual configuration and child-job bills.

Adding a range for `batch_size` does nothing useful unless the script parses it and actually trains with minibatches. More optimizer steps per epoch also changes runtime, so revisit the epoch choices and estimate.

First trial completed: `GloegeIan-nn-tune-261125-1128`; child `GloegeIan-nn-tune-261125-1128-001-8fc21bb9`. Recorded validation accuracy was 0.7413, wall time 3.6 minutes, and billable time 2.9 minutes (about $0.005 at the lesson rate). Do not relaunch it. No 10-trial search is recorded as launched. After tuning, evaluate the winner once on untouched test data; validation selection and final test assessment serve different purposes.

## 9. Fine-tuning with SageMaker

Use an appropriate pretrained model first. LoRA freezes base weights and trains small adapter matrices; this greatly reduces optimizer and gradient memory, but the whole model still participates in computation. Full fine-tuning requires much more memory and is not permitted in this workshop.

The lesson's class exercise adapts CHURRO-3B to historical German handwriting on one 24 GB `ml.g5.xlarge`. Its roughly half-hour runtime and sub-dollar cost are the lesson's measurements, not our results. Verify the current script, dataset, image, rate, and estimate before using them.

Plan memory from base weights, trainable adapters, optimizer states, activations, and sequence/image tokens. Image resolution and batch size can dominate memory and runtime. Quantization support depends on the model, container, and training method; do not assume an inference configuration supports training unchanged.

Keep the base model revision fixed and save adapters, processor/tokenizer configuration, hyperparameters, split manifest, evaluation outputs, and training history. Save epoch checkpoints to `/opt/ml/checkpoints` with an explicit S3 checkpoint prefix. A new experiment gets its own prefix; intentionally resuming an experiment uses its existing checkpoints. The script must actually load and resume training state—S3 synchronization alone does not implement resumption. [AWS checkpoint behavior](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html).

Measure the unchanged base and adapted model on the same held-out items using the task's own metric. For transcription use CER, where lower is better. Inspect examples and failure types as well as the aggregate. Hold out whole documents or groups when the real test contains unseen documents; a page-level split can be optimistic. Small evaluation samples are noisy. Read the held-out metric curve, not just decreasing training loss, and keep the best checkpoint within a declared epoch budget. Additional epochs, resolution experiments, and homework runs require new estimates and approval.

## 10. RAG, Bedrock, and processing jobs

Fine-tuning changes model behavior or capability; RAG supplies retrieved evidence at inference time. For document questions, build a reproducible corpus of page-aware chunks with document/page identifiers. Cache parsed text, chunks, embeddings, and index metadata in S3 so later comparisons reuse the same inputs.

Evaluate retrieval and generation separately. Keep the question set and scoring procedure fixed across model/backend comparisons. Inspect missing evidence, irrelevant retrieval, wrong answers, and citation failures. For tables and figures, compare text extraction with page-image or multimodal retrieval instead of assuming text chunks preserve the evidence.

Bedrock shifts hosting to a managed API and bills usage rather than an idle GPU. Use the lesson's approved models and tagged inference profiles, bound the number of embedding/generation calls and output tokens, estimate the whole exercise, and record returned usage. Bedrock has no instance type or training-job runtime cap: approvals should say that and instead specify models, request counts, token limits, retry limits, and expected cost. Do not hide API spend behind the absence of an instance.

Processing jobs are appropriate for finite batch inference on self-hosted weights. Stage S3 inputs, use container input/output paths, set `max_runtime_in_seconds`, tag the job, and monitor it. Preserve output URIs and evaluate only after completion. A GPU notebook is useful for an authorized interactive exercise but can bill while idle; stop it explicitly. The workshop does not require an always-on endpoint.

Reuse provided inputs instead of paying to recompute them. In episode 12 the instructor's `gte-large` embeddings were already staged in the shared bucket, so the notebook copied them (`use_instructor_embeddings = True`) and the embedding job was never launched. Check a flag like that before running a cell that can start a job: the episode 12 embedding cell launches only if the shared matrix is missing.

Episode 12 run, approved by Ian before launch:

| Item | Value |
|---|---|
| Job | `WattBot-GloegeIan-rag-1126-023847`, Completed |
| Hardware | 1 × `ml.g5.xlarge`, 60 GB volume, 3,600-second cap |
| Model | `Qwen/Qwen3-8B` with transformers, batch size 4, cached `gte-large` embeddings |
| Wall time | 14.7 min, of which 0.6 min provisioning was not billed |
| Billed | 14.0 min at the lesson rate of $1.41/h: **$0.33** (approved bound $1.42) |
| WattBot score (`Score.py`) | **0.5428** on the 53-question stratified sample |
| Components | answer value 0.4717 (weight 0.75), citation F1 0.7044 (0.20), clean refusals 0.9623 (0.05) |
| By evidence type | quote 17/29, table 6/12, figure 6/10, math 4/17, is_NA 3/5, cross-paper 3/15, reconcile 5/10 |

Math and cross-paper questions are the weak spots (about 20% correct), and 2 of the 5 unanswerable questions were answered anyway. The job ran in half the lesson's 15-minute estimate. A processing job bills from the end of provisioning to the end of the script, so the time cap bounds the cost even if the script hangs.

## 11. Troubleshooting and session recovery

- Distinguish a browser-control failure from an AWS failure. Errors such as `Debugger unattached`, CDP timeouts, or unavailable browser hosts do not prove a job failed.
- In this chat, a stale notebook tab consumed 6.3 GB of browser memory. Closing that browser tab and opening a fresh authenticated connection restored control. Saved notebook files and managed jobs were independent of that tab.
- If AWS sign-in expires, restore the UW SSO role through the established sign-in link. The user completes NetID/Duo. Never ask for passwords, PATs, or AWS keys in chat.
- Restore kernel variables from known setup cells, then attach to persisted job names. Do not use Run All on a partially completed notebook containing launch cells.
- Inspect `FailureReason`, secondary status, and CloudWatch logs before changing code. For an explicit deny, check tags, role, instance allowlist, runtime limit, and resource type. For quota or capacity errors, distinguish them from an invalid image or script failure.
- Pin compatible framework/Python/container versions and read current SDK documentation. The workshop uses SDK v2 classes; a deprecation warning alone is not a reason to migrate a working exercise mid-run.

### Driving Jupyter from a browser agent (learned 2026-11-26)

- **Never type into a cell unless it is verified to be in edit mode.** In command mode every key is a shortcut: `b` inserts a cell, `m` turns a cell into markdown, `d d` deletes, `x` cuts, `0 0` restarts the kernel. One line of code typed in command mode scrambled episode 12's open page and restarted its kernel. Safe sequence: select the cell, press Enter, then confirm the cursor is inside that cell's editor before typing.
- **Prefer keys that cannot edit:** Shift+Enter runs the selected cell and moves on, and the arrow keys move without running. A notebook can be rebuilt cell by cell this way, skipping launch cells with the down arrow.
- **Change setup values by editing the saved file, not by typing.** Episode 15's name, project and bucket were set by rewriting the notebook through the Jupyter contents API (`/api/contents/...`, PUT) before opening it.
- **When a page is damaged, back up the file at once.** Jupyter autosaves about every two minutes, and the damaged episode 12 page overwrote the good file within that window. A copy made through the contents API (POST with `copy_from`) right after the mistake allowed a full restore with a PUT. Then close the damaged tab without saving; a page with unsaved changes asks to confirm leaving, which only the user can answer.
- **Wait for the kernel to be idle before running cells.** Shift+Enter pressed while a freshly opened notebook's kernel was still starting ran nothing, silently.
- **Make launch cells safe to rerun.** Episode 12's launch cell asserts that its saved job-name file (`episode12-processing-job.txt`) does not exist, so a rerun stops instead of starting a second job. After a kernel restart, read the job name back from that file; in-memory references such as `rag_processor.latest_job` are gone.
- **Check job status from the notebook.** The AWS console needs a fresh SSO sign-in, which is the user's to do. `session.sagemaker_client.describe_processing_job(...)` from the controller works without it.
- **Read notebook state without the page.** `/api/contents/<path>` returns each cell's source, execution count and outputs; `/api/sessions` shows which kernels are running and idle. Browser tools may refuse output containing query strings or cookies, so strip those characters before returning text.

## 12. Cleanup and handoff

Before ending a session, enumerate this user's training, tuning, and processing jobs in the correct region, including all pages of list results. Verify terminal states; an unneeded job should be stopped explicitly and monitored until stopped. Do not touch other users' jobs in this shared account.

Stop `BadgerScribe-GloegeIan` and verify `Stopped`. Closing Jupyter is insufficient. A stopped notebook still has volume charges; S3 objects and other retained resources can also continue charging. Keep useful checkpoints, code, logs, and results, and seek approval before permanently deleting data or the notebook.

Episode 15 run, 2026-11-26 about 03:15 UTC (read-only listing cells; the delete cell ran with `confirm_delete = False`):

| Check | Result |
|---|---|
| Our notebook | `BadgerScribe-GloegeIan`, InService, `ml.t3.medium`; stopping it is a console step for Ian |
| Running jobs | 2 InProgress in the shared account, none with our name: nothing of ours is billing |
| S3 bucket | `badgerscribe-gloegeian`, 0.50 GB (about a cent a month) |
| Bedrock profiles | none with the `BadgerScribe-GloegeIan-` prefix, so nothing to delete |

After these checks, `BadgerScribe-GloegeIan` was stopped from the console on 2026-11-26 and verified `Stopped`. That finished the workshop. To use it again: sign in through the shared account's access-portal link (the lesson's form is `https://<org>.awsapps.com/start/#/console?account_id=...&role_name=...`, from the instructors) with your NetID, open SageMaker AI → Notebook instances in `us-east-2`, and click **Start** on `BadgerScribe-GloegeIan`. The 20 GB volume, the lesson clone and the notebooks are kept while it is stopped.

The account is shared, so listings include other learners' resources. Filter by our name and never stop or delete anyone else's resources.

Leave a handoff with completed and pending exercises, exact job names and statuses, artifact/checkpoint URIs, measured billable time, pricing assumptions, approvals already received, current notebook status, and any concrete blocker. Never report a submitted job as completed or the workshop as finished without verification.

## 13. MLM26 credit request and Marathon rules (2026-11-26)

After finishing the workshop, Ian submitted the AWS credit request for the team's Qwen3.8-27B fine-tuning on 2026-11-26.

**The exception.** The form's defaults are a $50 per-person budget, PEFT only on base models up to 7B, and models above 7B for inference only, through Bedrock. Chris Endemann, the Marathon admin, approved more for our team (Team Bucherwurm; confirmed with him 2026-11-26): LoRA or QLoRA fine-tuning of Qwen3.8-27B, up to $1,000 for training, on powerful instances if needed. The run that produces a competition submission must still fit in 96 GB of GPU memory. Training is not bound by that limit. The 27B model in bf16 is about 54 GB of weights, so it fits. Keep the approval on record with Chris whenever a form checkbox repeats the default limits.

**What the request said.**

| Field | Answer |
|---|---|
| Task | "Other": LoRA / QLoRA fine-tuning of Qwen3.8-27B, exception approved by Chris Endemann. The fine-tuning branch's size options stop at 7B, so it could not be answered truthfully |
| Dataset | Badger Scribe competition data, 11-20 release, about 2.1 GB (1,070 labelled training pages, 238 test pages); training uses training pages only |
| Instances | `ml.t3.medium` (controller); `ml.g6e.2xlarge` (1 × L40S 48 GB, about 64 GB RAM) for 4-bit QLoRA and inference; `ml.g6e.12xlarge` (4 × L40S, 192 GB) for bf16 LoRA |
| Trainable parameters | about 100 million at LoRA rank 16 (an estimate from layer sizes; confirm with `model.print_trainable_parameters()`) |

**Instance notes.** Prefer `ml.g6e.2xlarge` over `ml.g6e.xlarge` for the 27B model. The GPU is the same, but the xlarge has about 32 GB of system RAM, and 4-bit Qwen runs on EFabric used close to 60 GB of container RAM while loading and training. The 24 GB GPUs (`ml.g5.xlarge`, `ml.g6.xlarge`) cannot hold the 27B model with room to train, and `ml.g4dn.xlarge` has no bf16. Multi-GPU boxes are scarce on demand in November and December: another team's `ml.g6e.12xlarge` job waited hours for capacity before it ran. Chris's guidance was that QLoRA and LoRA give nearly the same fine-tuned model. QLoRA trains more slowly, because the weights are dequantized on every pass, but a single GPU is far easier to get.

**Rules accepted in the stewardship section.** These apply to all AWS use for the Marathon. The form's reason is cost and energy: high-powered or many instances use a lot of power, so test small, scale only when needed, pause what is idle and remove what is unused. (Its 7B and $50 limits are left out here; the exception above replaces them for this project.)

- **Estimate first.** Before any training or fine-tuning job, estimate its cost with the lesson's helpers (`helpers.estimate_finetune_cost()` from the fine-tuning episode).
- **Test before scaling.** Run new code locally or on a low-power notebook instance with a small data subset first, and use bigger hardware only once it works.
- **Start small.** Begin with the smallest instance type and count that can do the job, and scale up only if necessary.
- **At most 3 instances per job.** More instances multiply the cost. Before adding instances, test whether more of them actually speeds the job up. One `ml.g6e.12xlarge` is one instance with four GPUs.
- **Cap runtime.** Set `max_run` on every training and tuning job. GPU jobs are capped at 48 hours.
- **Tag everything.** Every bucket, notebook and job carries `Project`, `Name` and `Purpose`. Tags are required and case-sensitive, and a job with a missing or misspelled tag is refused at launch. Allowed Projects: WattBot2026, BadgerScribe, BadgerCode, BrainDecoding, PoisonedWell, SnapshotWI. Name is LastFirst. Purpose is Storage, Notebook, Training, Tuning or Inference. Ours: `Project=BadgerScribe`, `Name=GloegeIan`.
- **Track cost.** Check [AWS Cost Explorer](https://us-east-1.console.aws.amazon.com/costmanagement/home#/cost-explorer) regularly, filtering by tag and date range. Costs can take up to 48 hours to appear, so scale experiments up slowly and leave a couple of days between testing a newer or larger model and running it at scale.
- **Resubmit the form** for any new major task, including switching to a significantly larger model.
- **Launch from SageMaker notebooks only**, not the CLI or scripts on a laptop, so admins can see what was launched and from what code. This rules out an agent submitting AWS jobs from the Mac.
- **Long jobs:** launch with `wait=False` and stop the notebook instance while the job runs ([lesson guidance](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/08_Finetuning-BadgerScribe.html)).
- **Pause idle notebooks.** Regularly check for running notebook instances and stop any you are not working in.
- **Remove unused data.** Delete data that is no longer in use to free storage, during the project and not only at the end.
- **Clean up at the end.** When done with AWS, delete the buckets and notebook instances you created. Deletion is permanent, so list exactly what would go and get Ian's approval first.

## Sources and freshness

The course is evolving; cached pages can show older episode numbering, prices, and code. Reconcile the live lesson with the cloned script before launch. Local session observations above take precedence over generic example numbers as evidence of what actually ran.

- [Course index](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/index.html)
- [PyTorch and distributed comparisons](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/07_Training-models-in-SageMaker-notebooks-part2.html)
- [Hyperparameter tuning lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/08_Hyperparameter-tuning.html)
- [Fine-tuning lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/09_Finetuning-BadgerScribe.html)
- [Bedrock RAG lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/11_RAG-Bedrock.html)
- [RAG with Processing jobs lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/12_RAG-Processing-jobs.html)
- [Resource cleanup lesson](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/15_Resource-management-cleanup.html)
- [AWS notebook instances](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html)
- [AWS automatic model tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
- [AWS checkpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)

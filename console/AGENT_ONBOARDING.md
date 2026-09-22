# RunPod Media Console: Agent Onboarding and Handoff

Last updated: 2026-09-21. Workspace: D:\RunPodMediaConsoleWeb.

## Mandatory handoff discipline
Read this document before working. At the end of every task, update this document with changes, test results, deployment state, outstanding issues, and exact next steps. Update relevant project documentation and operational instructions as behavior changes. Distinguish local code, running local service, deployed worker image, and verified GPU output. Never report a completed fix based only on mocked tests or a healthy endpoint. Preserve user changes and untracked work. Do not write credentials into documentation or logs.

## Latest release: September 20 workflow upgrade (completed September 21)
Publication verified: GitHub origin/main is `55490b2d6265fa9c5d6010d3b44405dd48f3e372` on 2026-09-21. The GitHub-connected image build and GPU export are still unverified; source publication is complete. The nested worker checkout is clean. This local operational note records the published commit without triggering a second cloud rebuild.
This section supersedes older deployment instructions below. The user authorized committing and pushing this work. The root checkout has no remote; publish from `endpoint/minimax-h3-runpod` to its existing `origin/main`. That repository also distributes the Windows app in `console/`; keep shared files synchronized. GitHub/RunPod builds the image, so no local Docker installation is needed.

- Console version: `web-v15.82-workflow-variants`; local service restarted and /api/version confirmed on 2026-09-21. Workflow selector in **Inputs & Subjects** offers Legacy, FFLF September 20, and Ref2V September 20. Shared validation is `h3_workflow_options.py`; worker graph adaptation is `workflow_variants.py`; UI controls are `static/h3_variants.js`.
- Originals remain under `D:/Downloads2`. The checked-in `workflows/*_20260920.json` files are sanitized provenance/sigma manifests, NOT executable API graphs. The builder compiles the runtime graphs from the existing tested presets. Never publish the original saved prompts or media filenames. See `WORKFLOW_VARIANTS.md` in the worker checkout for API and graph details.
- Four photo slots, percentage-only keyframes, explicit multi-image bool, Larry/LightX switch, separate sigma choices 1/2/3/4, optional latent upscale and optional RTX output, last-frame PNG. Unused slots must have empty percentages. Photo 3/4 in FFLF require multi-image. With multi-image on, any occupied slot can be the first anchor. R2V video/audio references remain available with multi-image off; combining them with keyframes is rejected instead of silently dropping them.
- New payload tasks are `fl2v_20260920` / `r2v_20260920`, so old worker images reject the requests rather than silently executing Legacy. Keep this compatibility guard. SAMimate explicitly stays on Legacy.
- A single additional LoRA chain feeds both passes. Larry selects the dedicated sampler; LightX uses a normal sampler and a task-matched selectable LoRA. Latent upscale off removes all second-pass nodes/model requirements. RTX off removes the NVIDIA node requirement.
- Combined **Generation & models** tile groups sampling/resolution and model/encoder controls. Full title bars drag; left/right/center docking and persistence were tested in the browser. All 12 tabs have tile controls. Legacy image inputs are stored separately from the new photo slots.
- Worker dependencies are pinned in Dockerfile: H3 keyframes (only keyframe module loaded, no unrelated global patches), Minimax latent 3D upscaler, NVIDIA RTX nodes with nvidia-vfx 0.1.0.1. NVIDIA's package index lists Linux CPython 3.12 wheels. This is dependency compatibility evidence, not a successful GPU render.
- Existing volume `vgc3ky6r6y` received only the missing latent upscaler at `models/latent_upscale_models/minimax_h3_latent_upscaler_3d_bf16.safetensors`: 690,592,992 bytes. Full stored-byte SHA-256 verified as `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`. No new volume or GPU-pool change. Both path YAML and runtime resolver now include this category; missing models fail before graph submission.
- Validation: 68 console Python tests; 44 worker Python tests; all six console Node UI test scripts; JS syntax and Git whitespace checks. Browser verified all three docking zones, layout persistence, FFLF/Ref2V selection, second-pass disabling, and no JS errors. Read-only local ComfyUI object_info checks found the required custom nodes and valid link slots; dynamic input groups require their dotted keys.
- Deployment boundary: this release is distributed through origin/main and the connected RunPod build; confirm the remote SHA after pushing. A Git push is NOT proof that the RunPod image finished building or reached workers. No new workflow inference was performed, so GPU export/quality and RTX hardware behavior remain unverified. Inspect the GitHub-connected build and actual endpoint worker image, then run a short synthetic test before long jobs.
- Always update this document AND relevant API/deployment docs after further work, including exact commit, image/build state, tests, real worker evidence, remaining blockers, and next steps. Never replace unverified GPU status with a test-only completion claim.

## Original user request and investigation history
1. Retrieve actual RunPod worker logs and show whether requests reach the worker, what stage is running, and actual progress.
2. Investigate failures since the recent update; no successful exports reported by user.
3. Revisit timeouts based on real worker activity; handle OOM by cancelling affected work and safely restarting the affected worker where supported.
4. Verify models are read from the existing network volume, without creating another volume or downloading duplicate model copies.
5. Expose all supported sampler modes.
6. Movable section tiles across all tabs: drag the title bar left/right for half rows or center for full rows; preserve the existing console.
7. Maintain this detailed onboarding document for agent/model handoffs.

## Architecture and boundaries
- Local Python FastAPI backend: app.py. Frontend: static/app.js, static/index.html, static/styles.css.
- Local virtual environment: .venv. Launch via .venv\Scripts\python.exe -m uvicorn app:app. run_windows.bat uses 127.0.0.1:8765; inspect other launchers before restarting a live service.
- Output recovery: output_sync.py. Local outputs and SAM outputs live in workspace directories. Saved settings/history are generally under user profile .runpod_media_console_web; inspect paths in code and do not print secrets.
- Worker checkout: endpoint/minimax-h3-runpod. handler.py handles ComfyUI submission/history and export. This is a separate deployment boundary; inspect its Git state and any AGENTS.md before editing.
- Main repository contains console code; nested worker checkout may contain its own console copy. Determine canonical packaging/sync workflow before changing copies.
- Tests: tests/test_security_and_h3.py and other Python unittest files; tests/*.cjs use Node to verify UI logic. Worker tests are under endpoint/minimax-h3-runpod/tests.

## Historical facts from the initial investigation (superseded where noted below)
- H3 endpoint ID: s1e3i9book2zhh, name minimax-h3-runpod.
- Live endpoint image observed: registry.runpod.net/smiles4u17-minimax-h3-runpod-main-dockerfile:da86200bc.
- Live GPU pools observed: ADA_32_PRO and ADA_24, CUDA minimum 13.0, EU-RO-1, max workers 2, min 0. Recheck live; these are snapshots.
- Existing H3 volume historically vgc3ky6r6y mounted at /runpod-volume. Verify current attachment and model paths; do not provision replacements or modify GPU pools by assumption.
- User pasted failed job 7b1eb23b-dc63-4bd2-a4d5-3c50b5a70a6d-u1, worker vtnzzsd4b3svhj. Error: ComfyUI job exceeded 3600 seconds, executionTime 3601463.
- Submitted payload: Ref2V/r2v, 15 seconds, 1 megapixel, 20 steps, res_multistep/simple, Turbo disabled, reference image plus reference video; INT8 Ref2VA checkpoint, NVFP4/AWQ Qwen encoder, video/audio VAEs.
- Earlier submissions were rejected for FL2V LoRA on Ref2V and LightX2V combined with h3_turbo sampler. These are valid configuration protections, not proof of the subsequent timeout cause.
- Worker handler JOB_TIMEOUT_SECONDS defaults to 3600. _wait_for_history polls ComfyUI history and interrupts on timeout. Console timeout policy is separate; raising only the console limit does not raise the worker limit.
- Prior worker log retrieval returned only system/container lifecycle events, not generation-stage output. Runtime job status returned 404/job not found. Do not infer the exact slow/stalled stage from that.
- Previous progress percentage was elapsed-time-derived and capped at 95%; it was not sampling progress.

## Tools and skills
- Start RunPod work by reading C:\Users\test\.agents\skills\runpod\SKILL.md; debugging route: golden-paths/15-monitor-and-debug.md and runpod-mcp/SKILL.md.
- Discover tools with functions.exec and ALL_TOOLS. Structured RunPod MCP tools include list_endpoints, get_endpoint, endpoint_health, list_endpoint_workers, stream_worker_logs, get_job_status, get_template, and network-volume tools. Inspect current schemas/errors; do not assume parameter names.
- stream_worker_logs accepts endpointId, workerId, source, tail, since, maxWaitMs (not limit). Bound log reads. Redact secrets before showing/storing logs.
- Prefer structured RunPod tools for live infrastructure checks. runpodctl is fallback; inspect its --help and version. Read companion-clis skill for image/build/upload workflows if needed.
- For UI browser inspection use available browser automation; read computer-use skill when applying it. Native automation availability varies.
- Read applicable skills before using them. Use rg for targeted searches. Read memory registry C:\Users\test\.codex\memories\MEMORY.md for relevant prior work, then only the linked summaries needed. Do not edit memories unless explicitly requested; this onboarding file is project documentation, not a memory update.
- Do not spawn subagents unless user or applicable instructions explicitly request delegation.

## Operational safeguards and continuity
- Verify exact endpoint, worker, job, and volume before mutations. Avoid restarting a worker that may own another active job. Serverless workers differ from standalone pods; use supported lifecycle controls, not guessed pod IDs.
- Preserve separate per-endpoint network-volume settings and existing GPU configurations.
- Models should resolve under /runpod-volume/models; inspect actual loader paths, symlinks, startup download conditions, and mount visibility. File existence/size alone is not checksum validation.
- Store completed media under /runpod-volume/outputs where configured; trace worker output descriptors, upload/export, and local recovery separately.
- Do not rescan/download all historical outputs casually. Existing recovery uses an initial today-only scan and later durable checkpoints.
- Keep SAMimate masks and inverted previews reusable. Source-latent masked preservation is distinct from ordinary video reference input. Previous constraints included 24 FPS and 17k+5 aligned source/mask frames; verify implementation before changing.
- LightX2V files must match task: ref2v for r2v, fl2v for FL2V/text-only. Regular samplers such as Euler are required instead of h3_turbo. Preserve manually chosen steps unless selecting an explicit preset.
- Network-volume S3 requires authenticated uploads; presigned PUT compatibility has previously failed. Verify stored bytes for model installs; do not treat multipart completion or expected size as integrity proof.
- Beta5 replacement files in workspace represent separate prior work; do not alter/delete/restart them without inspection.

## Working-tree state at start of this task
Pre-existing untracked items included H3_WORKFLOW_AUDIT.md, docs/, beta5 replacement state and scripts, and scripts for H3 smoke testing / volume migration. Preserve them.
Changes made in preceding turn, still local:
- app.py: targeted ComfyUI timeout explanation.
- static/app.js: labels time-derived progress as estimated; replaces misleading finishing message with unknown ETA after estimate exceeded.
- tests/test_security_and_h3.py: timeout explanation regression test.
- tests/test_progress_ui.cjs: estimated-progress regression test.
Validation at that point: 44 Python tests passed, `test_progress_ui.cjs` and `test_h3_turbo_ui.cjs` passed, and `git diff --check` passed. Subsequent validation expanded this to 61 console Python tests and 36 worker Python tests; the local server was restarted and browser-checked. No worker deployment or GPU render verification was performed.

## Current task execution status
Onboarding was created first, as explicitly requested, and updated during this task. Local implementation is complete and verified. The live endpoint was read-only audited but not mutated or redeployed because RunPod's automatic approval returned a usage-limit block. The endpoint therefore still runs `da86200bc`; worker telemetry, volume-only startup protection, and the custom runtime inventory are local/nested-worker changes until a new image is built and deployed.

### New implementation details
- `runpod_monitor.py` reads RunPod v2 serverless worker logs as bounded JSON lines or SSE, validates IDs, preserves `workerId`, and redacts keys/URLs. `app.py` exposes `/api/endpoint/{endpoint}/workers/logs`, `/api/job/{endpoint}/{job}/logs`, `/diagnostics`, `/cancel`, and `/api/h3/sampling`.
- `endpoint/minimax-h3-runpod/telemetry.py` connects to ComfyUI WebSocket before submission, records node/stage/sampling steps, writes bounded diagnostics under `/runpod-volume/diagnostics/h3/<job>.json`, and emits RunPod progress updates. It has separate maximum-runtime and inactivity limits; logs and actual WebSocket progress reset inactivity. OOM and dead-ComfyUI paths interrupt, preserve diagnostics, return `refresh_worker`, and never silently resubmit.
- `endpoint/minimax-h3-runpod/custom_nodes/h3_runtime` reports the active GPU, CUDA capability, Comfy model search paths, sampler catalog, and volume mount. It resolves model names only inside the mounted `/runpod-volume`; missing/off-volume files fail closed. `start.sh` requires the network-volume mount and sets offline Hugging Face/Transformers mode, so the worker cannot silently download a replacement model.
- `h3_sampling.json` is pinned to the worker's ComfyUI revision and contains 46 sampler names plus the nine supported schedulers. The console loads the catalog at runtime and preserves visible selections.
- `static/dashboard.js` and `static/workspace.css` add saved, draggable section cards inside every existing tab. Cards can be ordered, collapsed, or toggled between half-width and full-row. The original top tab bar and bottom console log remain intact. `static/workspace.js` adds the Run Monitor and configurable H3 maximum/inactivity limits.

### Live evidence captured
- Endpoint `s1e3i9book2zhh` remains attached to network volume `vgc3ky6r6y` (`STANDARD`, 140 GB, `EU-RO-1`). No new volume was created.
- HEAD checks on the existing volume confirmed the selected pruned FL2VA and Ref2VA diffusion models (20,970,379,616 bytes each), NVFP4 text encoder (15,687,142,551 bytes), video VAE (5,207,808,496 bytes), and audio VAE (605,254,808 bytes).
- The live worker pool reported RTX 4090 workers. The NVFP4 encoder is present on the volume but is a Blackwell-targeted artifact; do not claim native NVFP4 performance or an OOM root cause on RTX 4090 without a deployed diagnostic run. The universal INT8 encoder was not found in the inspected volume listing.
- Recent local job history contains repeated ComfyUI connection refused/reset failures and replacement process exit `-6`, plus the earlier 3,600-second timeout. These establish worker instability but do not prove OOM. RunPod worker-log SSE calls timed out during this turn; the new monitor preserves durable worker-side diagnostics for the next run.
- Local service was running at `http://127.0.0.1:8765` with version `web-v15.65-worker-observability` at the time of that earlier inspection; it was restarted after the Pod-log change and now reports `web-v15.78-pod-logs`.

### Validation for this task
- Console Python: 61 tests passed.
- Worker Python: 36 tests passed.
- UI: `node --check` passed for app/workspace/dashboard scripts; H3 Turbo UI, progress wording, sampler/monitor tests passed.
- Browser verification: every tab had initialized section cards (`browser` 3, `h3` 7, `image_editor` 1, `inf` 8, `model_manager` 2, `monitor` 3, `sam` 3, `samimate` 9, `settings` 4, `tts` 3, `wan` 7, `workflow` 6). The H3 tab visibly showed draggable controls and full-row controls.
- No paid inference, worker image build, endpoint update, worker restart, volume write, model download, or output-quality render was performed in this task.

## Final handoff checklist (must update on completion or interruption)
- Files changed and purpose.
- Verified root causes, with job/worker IDs and dated evidence.
- Tests run and results; distinguish mocked from real GPU validation.
- Local service state and deployed image/build/revision.
- Model volume attachment and resolved loader paths; duplicate-download behavior.
- Log/progress transport, timeout policy, cancellation/recovery behavior.
- Sampler list source and UI layout controls.
- Remaining risks, blocked items, and exact next action.

### Next handoff action
Push worker commit `f924f11` from `endpoint/minimax-h3-runpod` to `origin/main` when external GitHub authorization is available. The worker suite passed 36 tests before the commit. After the push, wait for the RunPod GitHub build, verify the build/release on endpoint `s1e3i9book2zhh`, preserve `vgc3ky6r6y`, EU-RO-1, GPU pools, and existing model paths, then run one short diagnostic or 1–5 second H3 job. Watch the new Run Monitor and inspect the durable model-path inventory before attempting a longer export. Update this file again with the build ID, image tag, deployment timestamp, worker log lines, actual progress stages, and output artifact path.

### Deployment attempt: 2026-09-20
The tested worker changes were committed locally as `f924f11` (`Add worker observability and volume-only model runtime`). `git push origin main` was attempted but automatic approval rejected the external action because the account usage limit was reached; the push was not executed. The RunPod endpoint remains on the prior deployed image until that commit reaches GitHub and its connected build completes.

### UI drag fix: 2026-09-20
The first dashboard layout pass exposed only native HTML5 draggable buttons; Chromium/browser button dragging did not reliably produce drop events. `static/dashboard.js` now uses pointer dragging with a placeholder and saved order, plus a click-to-pick/click-to-drop fallback. `static/workspace.css` gives the grip a visible blue treatment and picked/drop states. The behavior is loaded for all 12 tabs; current handle counts are workflow 6, inf 8, image_editor 1, tts 3, wan 7, h3 7, model_manager 2, samimate 9, sam 3, settings 4, browser 3, monitor 3. `node --check static/dashboard.js` and `git diff --check` pass. The local browser verified a real reorder and persisted layout state.

## In-progress update: 2026-09-19 (new work not yet deployed)
User clarified layout: movable/collapsible SECTION cards inside every existing tab, half-row or full-row widths, drag/drop order; preserve existing console. Initial two-work-area interpretation was discarded.
Implemented locally so far: worker telemetry.py (Comfy WebSocket progress, RunPod progress updates, bounded durable job logs on volume, idle and max limits), volume-only loader custom node h3_runtime, startup mount guard/offline mode, OOM/crash refresh request, full pinned 46-sampler catalog, console log/diagnostic/cancel APIs, section-card layout in static/dashboard.js and monitor integration in static/workspace.js. Validation ongoing; do not deploy without remaining tests/review.
Live inspection: endpoint v34 da86200bc attaches vgc3ky6r6y, 140 GB, EU-RO-1. Both selected pruned INT8 diffusion models (20,970,379,616 bytes each), NVFP4 encoder (15,687,142,551 bytes), video VAE (5,207,808,496), audio VAE (605,254,808) HEAD-confirmed. Only NVFP4 encoder exists; universal INT8 encoder is absent. Failing worker is RTX 4090; NVFP4 falls back to dequantized compute on Ada according to Comfy-Org docs, so native Blackwell efficiency cannot be assumed. No model or volume downloads/creation performed. Do not claim this proves the exact crash cause.
History confirms repeated ComfyUI connection refused/reset and process restart exit -6 on Sep 19, plus one-hour timeouts. Container log API timed out twice. Current RunPod health has no active/queued jobs at last check.
Earlier running-service snapshot (PID 32204 parent / 16232 child) is historical. The service was restarted after the Pod-log change; recheck the current owner of port 8765 before stopping it again.

## Endpoint worker log integration: 2026-09-20
- The requested resource is the **serverless endpoint's workers**, not a separately launched ComfyUI Pod. `runpod_monitor.py` now reads RunPod REST v2 `serverless/{endpoint}/workers` discovery followed by `serverless/{endpoint}/workers/{worker}/logs`, accepts JSON lines or SSE, preserves each record's `workerId`, supports all workers or a pinned worker, and redacts keys/URLs.
- `app.py` exposes `GET /api/endpoint/{endpoint_id}/workers/logs?worker_id=&source=both&tail=200`. It validates the endpoint against configured/recorded jobs, retains recent lines through brief provider failures, reports the worker IDs observed in the stream, and keeps the RunPod key server-side.
- The Run monitor now has an **Endpoint worker logs** section with all-worker or pinned-worker selection, `container`/`system`/`both` filters, refresh, and a separate endpoint-worker output pane. Selecting a job pins the discovered worker when available; leaving the selector at all workers captures cold starts, crashes, and workers that receive another queued job.
- Validation for this addition: 67 console Python tests passed, including endpoint worker JSON-line/SSE parsing and route tests; `node --check static/workspace.js` passed; H3 Turbo/progress UI tests passed; `git diff --check` passed. These are local/mocked transport checks, not proof that the live endpoint currently has a running worker or that a GPU export succeeds.
- Local code version is `web-v15.79-endpoint-worker-logs`; static cache query is `15.79`. The local service was restarted to this version. The RunPod endpoint image is unchanged until the GitHub push/build completes. No worker was restarted, no volume was created, and no model was downloaded by this change.

### Handoff for endpoint-worker log follow-up
1. Open Run monitor and select a recent job. Inspect **Container + system** in **Endpoint worker logs**; system lines show worker startup/image lifecycle, while container lines show handler/ComfyUI output.
2. Leave the worker selector at **All endpoint workers** when diagnosing queueing or scale-up. Pin the worker ID only when correlating one job's crash or OOM.
3. Correlate worker timestamps with the job ID, durable diagnostics, and the endpoint health/worker counts. An empty response after the bounded read is inconclusive until a fresh worker timestamp is captured; RunPod can return no bytes when no matching worker line exists.
4. If logs show OOM or a dead handler, cancel the affected serverless job first, preserve the worker/system lines, and only then decide whether the endpoint needs a worker refresh or image change. Update this file with endpoint ID, worker ID, timestamps, log evidence, and the result.

### Verified endpoint log correction, 2026-09-20
Live reads now return actual endpoint worker logs. RunPod rejects source=both with HTTP 422 (only container/system allowed); the helper reads both sources separately and combines them. Each record is tagged with the worker ID from its request. Reads retain received lines when an idle stream times out, and successful workers survive errors from another worker. All-worker selection remains selected across polling.
Live discovery returned five workers; a bounded read returned 20 system records including worker ready, stop/remove container, and image tag f924f11dd on worker wrx97a048ve9tr at 2026-09-20T09:14:39Z. This supersedes the earlier old-image deployment snapshot: the new image has reached workers. It does not establish a successful generation or prove the exact export failure. No infrastructure mutation or inference was performed. Console restarted on port 8765 with corrected transport. 67 Python tests and workspace JavaScript syntax passed.

## Active task: September 20 workflow upgrade
User supplied NEW FFLF 9_20_26.json and Minimax_H3_Ref2V UPSCALE DECENT 9_20_26.json in D:/Downloads2. Add selectable variants retaining Legacy; four photo slots with percentage-only keyframes and explicit multi-image boolean; sigma choices 1/2/3 = 3/4/5 steps, 4 = remaining first-pass sigmas; selectable Larry vs LightX; optional latent upscale; final RTX and last-frame outputs. Merge Models/Turbo with generation settings; make whole title bars draggable with left/right/center drop zones across tabs. User authorizes commit and push. Preserve unrelated model transfer state. Root repository has no remote; nested endpoint/minimax-h3-runpod owns origin and console/ distribution. Do not include saved prompts/private local media in published workflow examples.

## Gap-filling tile layout: 2026-09-21
Tiles now use height-aware dense grid placement on every tab. A later half-width tile can fill available space beneath a shorter tile instead of waiting for the taller neighboring tile to finish. Explicit left/right sides and full-width tiles remain respected; a tile only fills a gap large enough for its width and height. ResizeObserver remeasures after collapse/expand, media loading and viewport changes. Mobile stays a single natural-height column. Packing includes the drag placeholder and excludes the floating dragged tile. Static cache versions for dashboard.js and workspace.css are 15.83; refresh the browser to load them. No backend restart or worker change is required.
Validation: browser confirmed LoRAs moved directly below the short left tile alongside a tall right tile, reflowed on expansion, had no overlapping cards, and reverted to a single column at 700px. No browser JavaScript errors; node --check and git diff --check passed. Preserve these behaviors and update this handoff after future layout changes.

## Upscale control discoverability: 2026-09-21
The workflow selector now lives at the top of Generation & models, beside the upscale controls. Latent upscale & final output stays visible in Legacy with a disabled fieldset and an explanation to select a September 20 workflow. Labels distinguish Generation resolution / Pass 1 resolution from Latent upscale target (MP). Photo inputs remain in Inputs & Subjects. Browser confirmed the selector is in the models tile and the target resolution is visible in Legacy; no JS errors. JS syntax and whitespace checks passed. Cache version for h3_variants.js is 15.84. No worker/backend changes.

## Resolution labels: 2026-09-21
Generation & models now shows a live Requested sizes summary: original MP and target pixel count, optional latent upscale target, and final RTX long edge or the un-upscaled output stage. Labels explicitly identify Pass 1 original size and Pass 2 latent target. The hint distinguishes 0.20 MP (200,000 pixels) from 0.02 MP (20,000 pixels, below supported minimum). Requested targets are not presented as measured historical export dimensions. This UI change does not diagnose or alter previous outputs. Static h3_variants.js cache version is 15.85. Syntax and whitespace checks passed. Update handoff with any future export-dimension investigation.

## Simplified resolution/output controls: 2026-09-21
Supersedes the verbose resolution summary above: user clarified .02 was a typo. Removed pixel counts/explanatory summary. Labels are MP and Upscale MP; first pass defaults 0.2, latent target now defaults 1.0, both editable (latent target active in latent mode). Three exclusive output choices: 1 Not scaled (generation MP is final); 2 Latent upscaled (second pass to target MP); 3 RTX upscale (RTX directly from generation, no latent pass). UI persists output_mode, which shared validation maps to canonical booleans; legacy API requests without mode retain compatibility. Advanced JSON cannot override this UI field. Existing saved MP values are preserved; selecting a new workflow initializes 0.2/1 defaults. Cache 15.86. Validation: 45 worker tests including all three graph routes and 1 MP default; 68 console tests; JS syntax and whitespace checks. No GPU render performed for this change. Update this document on further handoff.

## RTX ordering correction: 2026-09-21
User clarified mode 3 includes BOTH upscale stages: generation MP -> latent Upscale MP / second pass -> RTX (1920x1080 at 16:9). This supersedes the preceding statement that mode 3 skips latent upscale. Shared output_mode validation now enables both flags for rtx; the UI keeps Upscale MP and second-pass controls editable in modes 2 and 3. Short labels and 0.2/1 MP defaults are unchanged. Regression checks assert the complete second-pass -> decode -> RTX chain and 1920x1080 dimensions. Static cache version 15.87. Keep this ordering when updating code/docs and update onboarding at handoff.

## Task-derived H3 controls and photo drops: 2026-09-21
This supersedes the workflow/output dropdown sections above. The H3 task determines the dated workflow (FL2V -> FFLF, R2V -> Ref2V); T2V and SAMimate retain legacy routing. No workflow or output-mode dropdown is shown. Use latent upscale controls only the latent/second-pass branch. RTX defaults on and is independent: it runs AFTER the latent pass when enabled, or after original decode when latent is disabled. New UI requests omit output_mode; old API clients retain compatibility.
LOWRES SIZE (MP), default 0.2, and FINAL SIZE (MP), default 1.0, sit beside each other with aspect ratio and duration. Final MP controls latent size and is disabled when latent is off. Sampling keeps steps/sigma controls together. No Turbo disables the actual Turbo LoRA node and dedicated sampler, while the Larry/LightX switch and selectable LoRA/strength sit together. Other user LoRAs remain separate. Root app.py and worker handler.py must both honor turbo_enabled; do not reintroduce forced True. Shared workflow_options.py carries this boolean.
All four photo cards share drag/drop upload binding, picker, preview and Clear (clears percentage too). Drops prevent browser navigation and stop bubbling to tile controls. Existing saved photo paths restore; older image paths migrate into slots. SAMimate cloned controls explicitly remove inherited hidden class. Saved split defaults are bounded below saved steps during migration.
Static cache is 15.88 for app.js, h3_variants.js and workspace.css. Validation: 68 console and 46 worker tests passed, including No Turbo with both families/tasks and RTX without latent. Seven JavaScript UI test scripts passed, including file-drop routing/prevention. Browser verified both MP fields share a row, switches control disabled state, four photo cards, and no JS errors. No paid GPU render or worker build completion verified. Commit/push through nested endpoint/minimax-h3-runpod origin; keep console/ copies synchronized. Every subsequent agent MUST read and update this onboarding, relevant workflow docs, and verification/deployment status before handoff.

## H3 cancel and tile hold gestures: 2026-09-22
Cancel current job sits beside Run MiniMax H3. static/h3_job.js tracks the latest H3 submission independently of monitor selection, disables duplicate cancel requests, permits retry on failure, and ignores stale completion callbacks from older jobs. It sends the existing POST /api/job/{endpoint}/{job}/cancel; polling confirms terminal state. On console-ready, restore the tracked job or check the latest H3 history submission and resume watching only if nonterminal. No actual job was cancelled in validation.
All tabs now use click title bar to expand/collapse and a 300 ms hold to start pointer dragging. Removed click-to-pick fallback. Left/right/center sizing and dense packing remain. Header action buttons remain independent; Escape/pointer cancel/window blur abort dragging. Browser verified click toggles details open state without JS errors. Automated UI checks cover delayed drag activation, post-hold click suppression, cancellation retries, duplicate suppression, and overlapping-job races. Nine UI test scripts and JS syntax/whitespace checks pass. Static app.js/dashboard.js and new h3_job.js cache version 15.89; browser refresh required, no backend restart. Root and nested console copies synchronized and committed locally (nested 8a61d5e, root 6506b6f). GitHub push was BLOCKED by automatic approval review: destination ownership/trust not established. Ask user to confirm smiles4u17/minimax-h3-runpod as the authorized destination before retrying; do not claim this change is pushed. Update onboarding and validation/release status on every handoff.

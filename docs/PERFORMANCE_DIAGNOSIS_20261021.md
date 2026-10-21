# H3 slow-generation diagnosis — October 21, 2026

Live evidence was captured at approximately 14:27–14:35 America/Los_Angeles.
Raw diagnostics and request records remain private under ignored `tmp/`.

## What the jobs actually did

The morning batch contains 12 submissions around 09:54–10:00 local time.
Eight have durable `completed` records; one is executing and three are queued.
The active job waited 16,225,470 ms (4h 30m 25s) before execution. This is queue
waiting, separate from sampling time. Health reported one running worker, no
unhealthy/throttled workers, one active job and three queued jobs.

All eight completed samples used RTX PRO 4500 Blackwell, Sage attention,
NVFP4 text encoding, the INT8 beta5 hybrid, balanced MLP chunking with prefetch,
8 LCM/simple first-pass steps, and no cache. The Sage production path passed all
four fp16/bf16 numerical startup checks (relative RMSE approximately 0.036–0.037).
This confirms the adaptive worker update took effect; these jobs are not using
the old native-attention worker.

Requests target 15-second videos, 0.2 MP first pass, latent upscale to 1 MP,
another eight clean low-noise refinement steps (second_pass_sigma=5), then RTX
upscale. The UI's 0.2 MP value therefore does not describe the expensive pass.
Eight diagnostics show 1,414–1,495 seconds total (mean 1,447.375 seconds,
24m 7s). Their refinement progress bars show 1,000–1,027 seconds (mean 1,012.5
seconds, 16m 53s), about 70% of total execution. Refinement steps take about
125–129 seconds each. This is observed execution cost, not an inferred Sage
fallback or a GPU hardware failure.

With one active worker these costs add across a batch. The selected eight-step
refinement is the largest verified rendering bottleneck. No duration, model,
LoRA, seed, sampler, resolution, steps, cache, GPU selection or scaling settings
were changed, and no original jobs were cancelled, retried or duplicated.
A lower refinement count or opt-in Sol recipe requires a quality comparison;
there is no measured speed/quality claim for either in this investigation.

There is also an approximately 78-minute gap between the last retained completion
(13:09 local) and the current execution starting around 14:27. Its cause is
unresolved: control-plane endpoint/worker-list reads fail with HTTP 500 and
`authentication failed`, including one retry and a read with the app credential.
Runtime health/status and volume diagnostics still work. This limits confirmation
of worker acquisition, lifecycle and deployment. Do not treat this API-side
response as proof that the user's credential is invalid, or tune blindly.

## Fixes

1. Bulk request lists can expire terminal records. Two completed jobs were shown
   as UNAVAILABLE even though diagnostics and every exact output size verified.
   The console now recovers a bounded recent set using diagnostics and HEAD
   metadata only, caches outcomes, reports completion source/execution time,
   and keeps Cancel disabled. It never downloads media during status listing.
   Live verification recovered both jobs after restarting only console port8765.
2. The SDK sends progress updates as asynchronous IN_PROGRESS result packets.
   Our previous completed/failed telemetry emitted another such packet at job
   shutdown. Terminal snapshots now remain in durable diagnostics/stdout while
   the SDK final result owns terminal status. This prevents those extra terminal
   progress packets; it is not claimed to explain the observed 78-minute gap.
3. Durable diagnostics now retain started_at, finished_at and cumulative per-node
   wall time/calls even after the rolling event list fills. Repeated unchanged
   stages do not force storage writes. These node timings include loading/waiting
   while a node is active; they are not CUDA profiler timings. The monitor shows
   them with effective generation metadata.
4. Corrected the regression suite's fixture-path isolation for every security
   test. Two known fake endpoint history rows were archived then removed; real
   history rows were retained.

Validation: 112 console Python tests, 84 worker Python tests and all 14 Node UI
scripts passed. JavaScript syntax and scoped Git whitespace checks pass.
Local console reports web-v16.15-durable-job-status. No Dockerfile changes or
local Docker operations were made. Source publication and cloud deployment
confirmation are separate: current control-plane errors prevent verifying
managed build/release state from those APIs.

Source for the progress behavior:
https://github.com/runpod/runpod-python/blob/main/runpod/serverless/modules/rp_progress.py

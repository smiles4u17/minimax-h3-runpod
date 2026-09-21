/* Keep existing tabs and console; arrange the sections within each tab. */
(() => {
  const main=document.querySelector('main');
  const monitor=document.createElement('section');monitor.id='monitor';monitor.className='tab panel';
  monitor.innerHTML=`<h2>Run monitor</h2><div class="card"><h3>Job controls</h3><div class="monitorControls"><label>Job<select id="monitor_job"><option value="">Select a recent job</option></select></label><button id="monitor_history" type="button">Load history</button><button id="monitor_refresh" type="button">Refresh logs</button><button id="monitor_cancel" type="button" disabled>Cancel job</button><a id="monitor_runpod" target="_blank" rel="noopener">Open RunPod</a></div><p id="monitor_status" role="status">Choose a job to inspect its worker.</p><progress id="monitor_progress" max="100" hidden></progress><p id="monitor_stage"></p></div><div class="card"><h3>Endpoint worker logs</h3><div class="monitorControls"><label>Worker<select id="monitor_worker"><option value="">All endpoint workers</option></select></label><label>Log source<select id="monitor_worker_source"><option value="both">Container + system</option><option value="container">Container only</option><option value="system">System only</option></select></label><button id="monitor_worker_refresh" type="button">Refresh worker logs</button></div><p id="monitor_worker_status" role="status">Select a job, or load endpoint history, to inspect the workers that actually ran it.</p></div><div class="card"><h3>Worker output</h3><pre id="monitor_logs">No logs loaded.</pre></div><div class="card"><h3>Endpoint worker output</h3><pre id="monitor_worker_logs">No endpoint worker logs loaded.</pre></div><div class="card"><h3>Model storage & runtime details</h3><pre id="monitor_details"></pre></div>`;
  main.appendChild(monitor);
  const mb=document.createElement('button');mb.dataset.tab='monitor';mb.textContent='Run monitor';mb.onclick=()=>showTab('monitor');document.querySelector('header .tabs').appendChild(mb);
  const sections=new Map([...main.querySelectorAll(':scope > .tab')].map(s=>[s.id,s]));
  const jobs=new Map();let busy=false,selected='',lastLogs=0,workerBusy=false,selectedWorker='',lastWorkerLogs=0,currentWorkerEndpoint='';
  function addJob(endpoint,job,target){if(!endpoint||!job)return;const id=endpoint+':'+job;if(!jobs.has(id)){jobs.set(id,{endpoint,job,target});const opt=document.createElement('option');opt.value=id;opt.textContent=`${target||'Job'} · ${job}`;document.getElementById('monitor_job').appendChild(opt)}return id}
  window.attachJobMonitor=(endpoint,job,target)=>{selected=addJob(endpoint,job,target);document.getElementById('monitor_job').value=selected;document.getElementById('monitor_cancel').disabled=false;lastLogs=0;void refreshMonitor()};
  async function history(){try{const r=await api('/api/jobs/history');for(const j of r.items||[])addJob(j.endpoint_id,j.job_id,j.target);if(!selected&&jobs.size){selected=jobs.keys().next().value;document.getElementById('monitor_job').value=selected;void refreshMonitor()}}catch(e){document.getElementById('monitor_status').textContent=e.message}}
  function addWorker(worker){if(!worker)return;const select=document.getElementById('monitor_worker');if(![...select.options].some(o=>o.value===worker)){select.add(new Option(worker,worker))}return worker}
  async function endpointWorkerLogs(force=false){const j=jobs.get(selected);const endpoint=currentWorkerEndpoint||j?.endpoint;if(!endpoint||workerBusy)return;if(!force&&Date.now()-lastWorkerLogs<5000)return;workerBusy=true;lastWorkerLogs=Date.now();try{const source=document.getElementById('monitor_worker_source').value||'both';const worker=selectedWorker;const query=new URLSearchParams({source,tail:'300'});if(worker)query.set('worker_id',worker);const r=await api(`/api/endpoint/${encodeURIComponent(endpoint)}/workers/logs?${query}`);if(endpoint!==(currentWorkerEndpoint||jobs.get(selected)?.endpoint)||worker!==selectedWorker||source!==document.getElementById('monitor_worker_source').value)return;for(const id of r.worker_ids||[])addWorker(id);const lines=r.items||[];document.getElementById('monitor_worker_logs').textContent=lines.length?lines.map(x=>`${x.ts} [${x.worker_id||'worker'}] [${x.source}] ${x.line}`).join('\n'):'No endpoint worker logs returned yet.';document.getElementById('monitor_worker_status').textContent=r.message||`Reading ${worker||'all endpoint workers'} from ${endpoint}`}catch(e){document.getElementById('monitor_worker_status').textContent='Endpoint worker logs unavailable: '+e.message}finally{workerBusy=false}}
  function display(d){
    const stage=document.getElementById('monitor_stage'),bar=document.getElementById('monitor_progress');
    if(!d||!d.stage){stage.textContent=d?.message||'Waiting for worker diagnostics';bar.hidden=true;return}
    const age=Math.max(0,Math.round((Date.now()-Date.parse(d.updated_at))/1000));
    stage.textContent=`${d.stage.replaceAll('_',' ')}${d.node?' · '+(d.node_name||'node '+d.node):''}${d.total_steps?' · step '+d.step+' / '+d.total_steps:''} · ${d.worker_id||'worker'} · updated ${age}s ago${d.oom?' · OUT OF MEMORY':''}${d.refresh_worker?' · worker refresh requested':''}`;
    bar.hidden=!d.total_steps;if(d.total_steps)bar.value=100*Number(d.step)/Number(d.total_steps);
    if(d.logs?.length)document.getElementById('monitor_logs').textContent=d.logs.join('\n');
    document.getElementById('monitor_details').textContent=JSON.stringify({models:d.models,capabilities:d.capabilities,max_runtime_seconds:d.max_seconds,idle_timeout_seconds:d.idle_seconds,inactive_seconds:d.inactive_seconds,error:d.error,events:d.events},null,2);
  }
  async function refreshMonitor(forceLogs=false){
    const j=jobs.get(selected);if(!j||busy)return;busy=true;
    document.getElementById('monitor_runpod').href='https://console.runpod.io/serverless/user/endpoint/'+encodeURIComponent(j.endpoint);
    try{
      const d=await api(`/api/job/${j.endpoint}/${j.job}/diagnostics`);display(d);currentWorkerEndpoint=j.endpoint;if(d.worker_id){addWorker(d.worker_id)}
      document.getElementById('monitor_cancel').disabled=['completed','failed','invalid_input','diagnostics_complete'].includes(d.stage);
      if(forceLogs||Date.now()-lastLogs>15000){lastLogs=Date.now();const r=await api(`/api/job/${j.endpoint}/${j.job}/logs`);if(r.items?.length){const provider=r.items.map(x=>`${x.ts} [${x.source}] ${x.line}`).join('\n');document.getElementById('monitor_logs').textContent=provider+(d.logs?.length?'\n\nCOMFYUI JOB LOG\n'+d.logs.join('\n'):'')}document.getElementById('monitor_status').textContent=r.message||'Connected to RunPod worker logs'}
      void endpointWorkerLogs(forceLogs);
    }catch(e){document.getElementById('monitor_status').textContent='Logs unavailable: '+e.message}finally{busy=false}
  }
  document.getElementById('monitor_job').onchange=e=>{selected=e.target.value;selectedWorker='';currentWorkerEndpoint='';lastLogs=0;lastWorkerLogs=0;document.getElementById('monitor_worker').replaceChildren(new Option('All endpoint workers',''));document.getElementById('monitor_logs').textContent='Loading…';document.getElementById('monitor_worker_logs').textContent='Loading endpoint worker logs…';void refreshMonitor(true)};
  document.getElementById('monitor_worker').onchange=e=>{selectedWorker=e.target.value;lastWorkerLogs=0;document.getElementById('monitor_worker_logs').textContent='Loading…';void endpointWorkerLogs(true)};
  document.getElementById('monitor_worker_source').onchange=()=>{lastWorkerLogs=0;void endpointWorkerLogs(true)};
  document.getElementById('monitor_refresh').onclick=()=>refreshMonitor(true);
  document.getElementById('monitor_worker_refresh').onclick=()=>endpointWorkerLogs(true);
  document.getElementById('monitor_history').onclick=history;
  document.getElementById('monitor_cancel').onclick=async()=>{const j=jobs.get(selected);if(!j)return;try{const r=await api(`/api/job/${j.endpoint}/${j.job}/cancel`,{method:'POST'});document.getElementById('monitor_status').textContent=r.message||'Cancellation sent to RunPod';document.getElementById('monitor_cancel').disabled=true}catch(e){document.getElementById('monitor_status').textContent=e.message}};
  setInterval(()=>{if(!document.hidden&&sections.get('monitor').classList.contains('active')){void refreshMonitor();void endpointWorkerLogs()}},5000);
  void history();

  // Load the full pinned runtime list, preserving saved/custom selections.
  async function samplers(){try{const data=await api('/api/h3/sampling');for(const prefix of ['h3','samimate_h3'])for(const kind of ['sampler','scheduler']){const el=document.getElementById(prefix+'_'+kind);if(!el)continue;const current=el.value;for(const value of data[kind+'s'])if(![...el.options].some(o=>o.value===value))el.add(new Option(value.replaceAll('_',' '),value));el.value=current}}catch(e){log('Sampler list unavailable: '+e.message)}}
  void samplers();
  // Independent maximum and inactivity limits apply to H3 and H3-backed SAMimate.
  const controls=document.createElement('div');controls.className='fields monitorTimeouts';
  controls.innerHTML='<label>Maximum run time (minutes)<input id="h3_max_minutes" type="number" min="1" max="1440" value="240"></label><label>No activity timeout (minutes)<input id="h3_idle_minutes" type="number" min="1" max="1440" value="30"></label><p class="hint">Model-loading logs and real progress reset inactivity. The maximum is a separate hard limit. OOM/crashed workers refresh after saving diagnostics.</p>';
  document.getElementById('h3_sampler').closest('.fields').after(controls);
  let limits={};try{limits=JSON.parse(localStorage.getItem('h3-run-limits')||'{}')}catch{}
  for(const [id,fallback] of [['h3_max_minutes',240],['h3_idle_minutes',30]]){document.getElementById(id).value=limits[id]||fallback;document.getElementById(id).onchange=()=>localStorage.setItem('h3-run-limits',JSON.stringify({h3_max_minutes:document.getElementById('h3_max_minutes').value,h3_idle_minutes:document.getElementById('h3_idle_minutes').value}))}
  function withLimits(p){return {...p,max_runtime_seconds:Number(document.getElementById('h3_max_minutes').value)*60,idle_timeout_seconds:Number(document.getElementById('h3_idle_minutes').value)*60}}
  const originalH3Settings=window.h3Settings;window.h3Settings=(...args)=>withLimits(originalH3Settings(...args));
  const originalPayload=window.currentPayload;window.currentPayload=(target,...args)=>{const p=originalPayload(target,...args);return target==='h3'?withLimits(p):p};
  const originalSam=window.samimateH3Payload;if(originalSam)window.samimateH3Payload=(...args)=>withLimits(originalSam(...args));
  window.addEventListener('console-ready',()=>{initializeDashboard();void samplers()},{once:true});
})();

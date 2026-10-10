/* Bulk request status keeps the list independent from the selected log viewer. */
(() => {
  let rows=[],busy=false,lastRefresh=0;
  const pending=new Set(),requested=new Set(),errors=new Map();
  const key=j=>j.endpoint_id+':'+j.job_id;
  const element=id=>document.getElementById(id);
  function render(){
    const list=element('monitor_request_list');if(!list)return;
    list.replaceChildren();
    if(!rows.length){list.textContent='No sent requests found.';return}
    for(const job of rows){
      const id=key(job),row=document.createElement('div');row.className='monitorRequestRow';
      const details=document.createElement('div');details.className='monitorRequestDetails';
      const title=document.createElement('strong');title.textContent=job.filename_prefix||job.target||job.task||'RunPod request';
      const identity=document.createElement('code');identity.textContent=job.job_id;
      const meta=document.createElement('small');meta.textContent=[job.endpoint_id,job.submitted_at?new Date(job.submitted_at).toLocaleString():'Submission time unavailable',job.duration!=null?job.duration+'s':'',job.seed!=null?'Seed '+job.seed:''].filter(Boolean).join(' · ');
      details.append(title,identity,meta);
      if(job.prompt){
        const prompt=document.createElement('details'),summary=document.createElement('summary'),full=document.createElement('p');
        prompt.className='monitorRequestPrompt';
        summary.textContent='Prompt: '+String(job.prompt).replace(/\s+/g,' ').slice(0,160)+(String(job.prompt).length>160?'…':'');
        full.textContent=job.prompt;prompt.append(summary,full);details.appendChild(prompt);
      }else{const unavailable=document.createElement('small');unavailable.textContent='Prompt not saved for this older request';details.appendChild(unavailable)}
      const settings=document.createElement('small');settings.textContent=[job.aspect_ratio,job.sampler,job.model,job.steps!=null?job.steps+' steps':'',job.megapixels!=null?job.megapixels+' MP':'',job.latent_upscale===false?'Latent upscale off':'',job.rtx_upscale===false?'RTX upscale off':''].filter(Boolean).join(' · ');if(settings.textContent)details.appendChild(settings);
      if(errors.has(id)){const error=document.createElement('small');error.setAttribute('role','alert');error.textContent='Cancel failed: '+errors.get(id);details.appendChild(error)}
      const state=document.createElement('span');state.className='monitorRequestState';
      state.textContent=pending.has(id)?'Cancelling…':requested.has(id)?'Cancellation requested':(job.status||'UNKNOWN').replaceAll('_',' ');
      const actions=document.createElement('div');actions.className='monitorRequestActions';
      const inspect=document.createElement('button');inspect.type='button';inspect.textContent='Inspect';inspect.onclick=()=>window.attachJobMonitor?.(job.endpoint_id,job.job_id,job.target||job.task);
      const cancel=document.createElement('button');cancel.type='button';cancel.textContent=pending.has(id)?'Cancelling…':requested.has(id)?'Cancellation requested':'Cancel';
      cancel.disabled=!job.can_cancel||pending.has(id)||requested.has(id);
      cancel.setAttribute('aria-label','Cancel request '+job.job_id);
      cancel.onclick=()=>cancelRequest(job);
      actions.append(inspect,cancel);row.append(details,state,actions);list.appendChild(row);
    }
  }
  async function cancelRequest(job){
    const id=key(job);if(!job.can_cancel||pending.has(id)||requested.has(id))return;
    pending.add(id);errors.delete(id);render();
    try{
      const result=await api(`/api/job/${encodeURIComponent(job.endpoint_id)}/${encodeURIComponent(job.job_id)}/cancel`,{method:'POST'});
      requested.add(id);job.can_cancel=false;
      element('monitor_request_status').textContent=job.job_id+': '+(result.message||'Cancellation sent to RunPod');
    }catch(error){errors.set(id,error.message);element('monitor_request_status').textContent=job.job_id+': '+error.message}
    finally{pending.delete(id);render();void refresh(true)}
  }
  async function refresh(force=false){
    if(busy||(!force&&Date.now()-lastRefresh<15000)||!element('monitor_request_list'))return;
    busy=true;const button=element('monitor_requests_refresh');button.disabled=true;
    try{
      const result=await api('/api/jobs/requests'+(force?'?force=true':''));rows=result.items||[];lastRefresh=Date.now();
      for(const job of rows)if(['COMPLETED','CANCELLED','FAILED','TIMED_OUT','UNAVAILABLE'].includes(job.status))requested.delete(key(job));
      render();element('monitor_request_status').textContent=(result.warnings||[]).join(' · ')||`${rows.length} sent requests · updated ${new Date().toLocaleTimeString()}`;
    }catch(error){element('monitor_request_status').textContent='Request list unavailable: '+error.message}
    finally{busy=false;button.disabled=false}
  }
  window.refreshSentRequests=refresh;
  function initialize(){
    const monitor=element('monitor');if(!monitor)return;
    const card=document.createElement('div');card.className='card';card.id='monitor_requests';
    card.innerHTML='<div class="cardTitleRow"><h3>Sent requests</h3><button id="monitor_requests_refresh" type="button">Refresh requests</button></div><p id="monitor_request_status" role="status">Loading sent requests…</p><div id="monitor_request_list" class="monitorRequestList">Loading…</div>';
    monitor.querySelector('h2').after(card);
    element('monitor_requests_refresh').onclick=()=>refresh(true);
    const attach=window.attachJobMonitor;
    window.attachJobMonitor=(endpoint,job,target)=>{attach?.(endpoint,job,target);void refresh(true)};
    const show=window.showTab;
    window.showTab=(id,...args)=>{const result=show(id,...args);if(id==='monitor')void refresh();return result};
    void refresh(true);
    setInterval(()=>{if(!document.hidden&&monitor.classList.contains('active'))void refresh()},15000);
  }
  // Insert before dashboard initialization so this card gets the normal tile controls.
  initialize();
})();

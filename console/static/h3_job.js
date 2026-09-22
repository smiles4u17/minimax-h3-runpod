/* Cancellation always targets the latest H3 submission, independent of monitor selection. */
(() => {
  const key='media-console-current-h3-job';let current=null;
  const button=()=>document.getElementById('h3_cancel_button');
  function render(){button().disabled=!!(!current||current.pending||current.sent);button().textContent=current?.pending?'Cancelling…':current?.sent?'Cancellation requested':'Cancel current job'}
  window.trackCurrentH3Job=(endpoint,job)=>{if(current?.endpoint===endpoint&&current?.job===job)return;current={endpoint,job};localStorage.setItem(key,JSON.stringify(current));render()};
  window.finishCurrentH3Job=(endpoint,job)=>{if(current?.endpoint!==endpoint||current?.job!==job)return;current=null;localStorage.removeItem(key);render()};
  window.cancelCurrentH3Job=async()=>{
    const selected=current;if(!selected||selected.pending||selected.sent)return;
    selected.pending=true;render();
    try{const r=await api(`/api/job/${encodeURIComponent(selected.endpoint)}/${encodeURIComponent(selected.job)}/cancel`,{method:'POST'});selected.sent=true;log(r.message||'H3 cancellation requested');if(current===selected)document.getElementById('h3_status').textContent='Cancellation requested';}
    catch(e){log('H3 cancellation failed: '+e.message);if(current===selected)document.getElementById('h3_status').textContent='Cancel failed: '+e.message}
    finally{selected.pending=false;render()}
  };
  window.addEventListener('console-ready',async()=>{
    let saved;try{saved=JSON.parse(localStorage.getItem(key)||'null')}catch{return}
    if(current)return;
    try{if(!saved?.endpoint||!saved?.job){const history=await api('/api/jobs/history');const latest=(history.items||[]).find(j=>j.target==='h3'&&j.endpoint_id&&j.job_id);if(!latest||current)return;saved={endpoint:latest.endpoint_id,job:latest.job_id}}
    const r=await api(`/api/job/${encodeURIComponent(saved.endpoint)}/${encodeURIComponent(saved.job)}`);if(current)return;if(['COMPLETED','FAILED','CANCELLED','TIMED_OUT'].includes(r.status.status)){localStorage.removeItem(key);return}watch(saved.endpoint,saved.job,'h3')}
    catch(e){log('Could not restore H3 job controls: '+e.message)}
  },{once:true});
})();


/* Dated workflow controls. Legacy stays the default and keeps its own inputs. */
(() => {
  const models=document.querySelector('[data-h3-section="models"]');
  const generation=document.querySelector('[data-h3-section="generation"]');
  models.querySelector('strong').textContent='Generation & models';
  models.querySelector('small').textContent='Resolution, sampling, models and upscale';
  models.open=true;
  const generationBody=generation.querySelector('.h3SectionBody');
  models.insertBefore(generationBody,models.querySelector('.h3SectionBody'));generation.remove();
  generationBody.insertAdjacentHTML('afterbegin','<h3>Sampling & resolution</h3>');
  models.querySelectorAll('.h3SectionBody')[1].insertAdjacentHTML('afterbegin','<h3>Models & encoders</h3>');
  const selector=document.createElement('label');
  selector.innerHTML='Workflow<select id="h3_workflow_variant"><option value="legacy">Legacy (original workflow)</option><option value="fflf_20260920">FFLF · September 20 · two pass</option><option value="ref2v_20260920">Ref2V · September 20 · two pass</option></select>';
  document.getElementById('h3_task').closest('.cardTitleRow').prepend(selector);
  const panel=document.createElement('div');panel.id='h3_variant_controls';panel.className='';
  panel.innerHTML=`<h3>Output</h3><p id="h3_variant_notice" class="hint"></p><fieldset class="fields" style="border:0;padding:0;margin:0;min-width:0">
    <label><input type="checkbox" id="h3_use_multi_image"> Use Multi IMG (percentage keyframes)</label>
    <label><input type="checkbox" id="h3_use_larry"> Use Larry Turbo (off = LightX2V)</label>
    <label>Output<select id="h3_output_mode"><option value="none">1 — Not scaled</option><option value="latent" selected>2 — Latent upscaled</option><option value="rtx">3 — RTX upscale</option></select></label><input type="checkbox" id="h3_latent_upscale" hidden><input type="checkbox" id="h3_rtx_upscale" hidden>
    <label>Upscale MP<input id="h3_final_megapixels" type="number" min="0.2" max="2" step="0.1" value="1"></label>
    <label>First-pass sigma split<input id="h3_pass1_split" type="number" min="1" value="3"></label>
    <label>Second-pass sigmas<select id="h3_second_pass_sigma"><option value="1">1 — 3 steps</option><option value="2" selected>2 — 4 steps</option><option value="3">3 — 5 steps</option><option value="4">4 — remaining first-pass sigmas</option></select></label>
    <label>Latent upscale model<input id="h3_latent_upscale_model" value="minimax_h3_latent_upscaler_3d_bf16.safetensors"></label>
    </fieldset>`;
  models.querySelector('.h3SectionBody').prepend(panel);
  const photos=document.createElement('div');photos.id='h3_variant_photos';photos.className='hidden';
  photos.innerHTML='<h3>Workflow photos</h3><p class="hint">Leave unused photo slots and their percentages empty. Photos 3 and 4 in FFLF require Use Multi IMG. Ref2V keeps the video/audio references below.</p><div class="fields">'+Array.from({length:4},(_,i)=>`<div class="variantPhoto"><label>Photo ${i+1}<input id="h3_photo${i+1}" placeholder="Local image path"></label><button type="button" data-photo="${i+1}">Choose image</button><input type="file" id="h3_photo_file${i+1}" accept="image/*" hidden><label>Keyframe percentage<input id="h3_keyframe${i+1}" placeholder="${i===0?'0%':i===1?'100%':'Leave empty if unused'}"></label></div>`).join('')+'</div>';
  selector.closest('.cardTitleRow').after(photos);
  generationBody.prepend(selector);
  photos.querySelectorAll('[data-photo]').forEach(button=>button.onclick=()=>document.getElementById('h3_photo_file'+button.dataset.photo).click());
  for(let i=1;i<=4;i++)document.getElementById('h3_photo_file'+i).onchange=async e=>{
    const file=e.target.files[0];if(!file)return;
    try{const form=new FormData();form.append('file',file);form.append('kind','image');const r=await fetch('/api/upload',{method:'POST',body:form});const data=await r.json();if(!r.ok)throw Error(data.detail||'Upload failed');set('h3_photo'+i,data.path);persistSettingsSoon()}catch(err){log(err.message)}
  };
  function isNew(){return val('h3_workflow_variant')!=='legacy'}
  function sync(change=false){
    const active=isNew(),fflf=val('h3_workflow_variant')==='fflf_20260920';
    panel.querySelector('fieldset').disabled=!active;photos.classList.toggle('hidden',!active);
    $('h3_variant_notice').textContent=active?'':'Select a September 20 workflow to enable upscaling.';
    setChk('h3_latent_upscale',['latent','rtx'].includes(val('h3_output_mode')));
    setChk('h3_rtx_upscale',val('h3_output_mode')==='rtx');
    const resolutionLabel=$('h3_megapixels').closest('label');
    for(const node of resolutionLabel.childNodes)if(node.nodeType===3&&node.textContent.trim()){node.textContent='MP';break}
    $('h3_task').disabled=active;
    if(active){set('h3_task',fflf?'fl2v':'r2v')}
    updateH3TaskUI();
    if(active)$('h3_fl2v_inputs').classList.add('hidden');
    $('h3_reference_grid').classList.toggle('hidden',active);
    for(const el of [$('h3_reference_grid').previousElementSibling,$('h3_reference_grid').previousElementSibling.previousElementSibling,document.querySelector('.h3SubjectBookBar')])el?.classList.toggle('hidden',active);
    if(change&&active){set('h3_pass1_split',fflf?3:6);set('h3_steps',fflf?4:8);set('h3_megapixels',.2);set('h3_final_megapixels',1);setChk('h3_turbo_enabled',true);selectH3TurboFamily(chk('h3_use_larry')?'larry':'lightx2v')}
    if(active){setChk('h3_turbo_enabled',true);set('h3_turbo_family',chk('h3_use_larry')?'larry':'lightx2v');updateH3TurboUI(false)}
    for(const id of ['h3_turbo_enabled','h3_turbo_family','h3_sampling_preset'])$(id).closest('label')?.classList.toggle('hidden',active);
    document.querySelector('[onclick="applyH3CleanBaseline()"]')?.classList.toggle('hidden',active);
    $('h3_sampler').disabled=active&&chk('h3_use_larry');
    for(const id of ['h3_final_megapixels','h3_pass1_split','h3_second_pass_sigma','h3_latent_upscale_model'])$(id).disabled=!chk('h3_latent_upscale');
  }
  $('h3_workflow_variant').onchange=()=>{sync(true);persistSettingsSoon()};
  $('h3_use_larry').onchange=()=>{setChk('h3_turbo_enabled',true);selectH3TurboFamily(chk('h3_use_larry')?'larry':'lightx2v');sync();persistSettingsSoon()};
  $('h3_output_mode').onchange=()=>{sync();persistSettingsSoon()};
  const original=h3Settings;
  window.h3Settings=()=>{const legacy=original();const settings={...legacy,legacy_inputs:{first_frame_path:legacy.first_frame_path,last_frame_path:legacy.last_frame_path,reference_paths:legacy.reference_paths,reference_subjects:legacy.reference_subjects},workflow_variant:val('h3_workflow_variant'),output_mode:val('h3_output_mode'),use_multi_image:chk('h3_use_multi_image'),use_larry:chk('h3_use_larry'),latent_upscale:chk('h3_latent_upscale'),rtx_upscale:chk('h3_rtx_upscale'),final_megapixels:Number(val('h3_final_megapixels')),pass1_split:Number(val('h3_pass1_split')),second_pass_sigma:Number(val('h3_second_pass_sigma')),latent_upscale_model:val('h3_latent_upscale_model'),photo_paths:[1,2,3,4].map(i=>val('h3_photo'+i).trim()),keyframe_positions:[1,2,3,4].map(i=>val('h3_keyframe'+i).trim())};
    if(isNew()){settings.task=settings.workflow_variant==='fflf_20260920'?'fl2v':'r2v';settings.first_frame_path=settings.use_multi_image?settings.photo_paths.find(Boolean)||'':settings.photo_paths[0];settings.last_frame_path=settings.photo_paths[1];settings.reference_paths=settings.photo_paths.filter(Boolean);settings.turbo_enabled=true;settings.turbo_family=settings.use_larry?'larry':'lightx2v'}
    return settings;
  };
  const apply=applyH3Settings;
  window.applyH3Settings=(h3={},storage={})=>{
    apply({...h3,...(h3.legacy_inputs||{})},storage);set('h3_workflow_variant',h3.workflow_variant||'legacy');
    for(const [name,defaultValue] of Object.entries({use_multi_image:false,use_larry:false,latent_upscale:true,rtx_upscale:true}))setChk('h3_'+name,h3[name]??defaultValue);
    for(const [name,defaultValue] of Object.entries({final_megapixels:1,pass1_split:3,second_pass_sigma:2,latent_upscale_model:'minimax_h3_latent_upscaler_3d_bf16.safetensors'}))set('h3_'+name,h3[name]??defaultValue);
    set('h3_output_mode',h3.output_mode||((h3.rtx_upscale===true)?'rtx':h3.latent_upscale===false?'none':'latent'));
    for(let i=1;i<=4;i++){set('h3_photo'+i,(h3.photo_paths||[])[i-1]||'');set('h3_keyframe'+i,(h3.keyframe_positions||[])[i-1]||'')}
    sync();
  };
  window.addEventListener('console-ready',()=>{applyH3Settings(SETTINGS.h3||{},SETTINGS.h3_storage||{});sync()},{once:true});
})();

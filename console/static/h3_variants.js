/* Task-derived routing; legacy remains available to API clients and SAMimate. */
function bindH3PhotoDrop(card,upload){
  card.ondragover=e=>{e.preventDefault();e.stopPropagation();card.classList.add('drag');if(e.dataTransfer)e.dataTransfer.dropEffect='copy'};
  card.ondragleave=()=>card.classList.remove('drag');
  card.ondrop=async e=>{e.preventDefault();e.stopPropagation();card.classList.remove('drag');await upload(e.dataTransfer?.files?.[0])};
}
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
  const panel=document.createElement('div');panel.id='h3_variant_controls';panel.className='';
  panel.innerHTML=`<h3>Output</h3><p id="h3_variant_notice" class="hint"></p><fieldset class="fields" style="border:0;padding:0;margin:0;min-width:0">
    <label><input type="checkbox" id="h3_use_multi_image"> Use Multi IMG (percentage keyframes)</label>
    <label><input type="checkbox" role="switch" id="h3_use_larry"> Larry Turbo (off = LightX2V)</label>
    <label><input type="checkbox" role="switch" id="h3_latent_upscale" checked> Use latent upscale</label>
    <label><input type="checkbox" role="switch" id="h3_rtx_upscale" checked> RTX upscale</label>
    <label><input type="checkbox" role="switch" id="h3_no_turbo"> No Turbo</label>
    <label>FINAL SIZE (MP)<input id="h3_final_megapixels" type="number" min="0.2" max="2" step="0.1" value="1"></label>
    <label>First-pass sigma split<input id="h3_pass1_split" type="number" min="1" value="3"></label>
    <label>Second-pass sigmas<select id="h3_second_pass_sigma"><option value="1">1 — 3 steps</option><option value="2" selected>2 — 4 steps</option><option value="3">3 — 5 steps</option><option value="4">4 — remaining first-pass sigmas</option></select></label>
    <label>Latent upscale model<input id="h3_latent_upscale_model" value="minimax_h3_latent_upscaler_3d_bf16.safetensors"></label>
    </fieldset>`;
  models.querySelector('.h3SectionBody').prepend(panel);
  const photos=document.createElement('div');photos.id='h3_variant_photos';photos.className='hidden';
  photos.innerHTML='<h3>Workflow photos</h3><p class="hint">Leave unused photo slots and their percentages empty. Photos 3 and 4 in FFLF require Use Multi IMG. Ref2V keeps the video/audio references below.</p><div class="fields">'+Array.from({length:4},(_,i)=>`<div class="variantPhoto"><label>Photo ${i+1}<input id="h3_photo${i+1}" placeholder="Local image path"></label><button type="button" data-photo="${i+1}">Choose image</button><input type="file" id="h3_photo_file${i+1}" accept="image/*" hidden><label>Keyframe percentage<input id="h3_keyframe${i+1}" placeholder="${i===0?'0%':i===1?'100%':'Leave empty if unused'}"></label></div>`).join('')+'</div>';
  $('h3_task').closest('.cardTitleRow').after(photos);
  for(let i=1;i<=4;i++){
    const path=$('h3_photo'+i),card=path.closest('.variantPhoto'),picker=$('h3_photo_file'+i);
    const preview=document.createElement('img');preview.className='variantPhotoPreview hidden';preview.alt='Photo '+i;preview.draggable=false;card.appendChild(preview);
    const refresh=async()=>{const saved=path.value.trim();preview.classList.toggle('hidden',!saved);if(!saved){preview.removeAttribute('src');return}try{const url=await previewUrlForPath(saved);if(path.value.trim()===saved)preview.src=url}catch(e){log('Photo preview: '+e.message)}};
    path.addEventListener('change',()=>{if(!path.value.trim())set('h3_keyframe'+i,'');void refresh();persistSettingsSoon()});
    card.querySelector('[data-photo]').onclick=()=>picker.click();
    const upload=async file=>{if(!file)return;if(file.type&&!file.type.startsWith('image/')){log('Choose an image for Photo '+i);return}try{await uploadFileToTarget(file,path.id,'image');await refresh()}catch(e){log('Photo upload: '+e.message)}};
    picker.onchange=async()=>{await upload(picker.files[0]);picker.value=''};
    bindH3PhotoDrop(card,upload);
    const clear=document.createElement('button');clear.type='button';clear.textContent='Clear';clear.onclick=()=>{path.value='';set('h3_keyframe'+i,'');void refresh();persistSettingsSoon()};picker.after(clear);
    path.refreshPreview=refresh;
  }
  function variant(){return val('h3_task')==='fl2v'?'fflf_20260920':val('h3_task')==='r2v'?'ref2v_20260920':'legacy'}
  function isNew(){return variant()!=='legacy'}
  const taskUI=updateH3TaskUI;
  window.updateH3TaskUI=()=>{taskUI();if(isNew())$('h3_fl2v_inputs').classList.add('hidden')};
  function sync(change=false){
    const active=isNew(),fflf=variant().startsWith('fflf');
    photos.classList.toggle('hidden',!active);
    $('h3_variant_notice').textContent=active?'':'Upscale stages are available for FFLF and Ref2V.';
    updateH3TaskUI();
    $('h3_reference_grid').classList.toggle('hidden',active);
    for(const el of [$('h3_reference_grid').previousElementSibling,$('h3_reference_grid').previousElementSibling.previousElementSibling,document.querySelector('.h3SubjectBookBar')])el?.classList.toggle('hidden',active);
    if(change&&active){set('h3_pass1_split',fflf?3:6);set('h3_steps',fflf?4:8);selectH3TurboFamily(chk('h3_use_larry')?'larry':'lightx2v')}
    setChk('h3_turbo_enabled',!chk('h3_no_turbo'));
    set('h3_turbo_family',chk('h3_use_larry')?'larry':'lightx2v');
    if(chk('h3_turbo_enabled')&&chk('h3_use_larry'))set('h3_sampler','h3_turbo');
    if(!chk('h3_use_larry')&&val('h3_sampler')==='h3_turbo')set('h3_sampler','euler');
    updateH3TurboUI(false);
    for(const id of ['h3_turbo_enabled','h3_turbo_family','h3_sampling_preset'])$(id).closest('label')?.classList.add('hidden');
    document.querySelector('[onclick="applyH3CleanBaseline()"]')?.classList.add('hidden');
    $('h3_use_larry').disabled=chk('h3_no_turbo');
    $('h3_sampler').disabled=chk('h3_turbo_enabled')&&chk('h3_use_larry');
    for(const id of ['h3_latent_upscale','h3_rtx_upscale','h3_use_multi_image'])$(id).disabled=!active;
    for(const id of ['h3_final_megapixels','h3_pass1_split','h3_second_pass_sigma','h3_latent_upscale_model'])$(id).disabled=!active||!chk('h3_latent_upscale');
  }
  $('h3_task').addEventListener('change',()=>{sync(true);persistSettingsSoon()});
  $('h3_use_larry').onchange=()=>{selectH3TurboFamily(chk('h3_use_larry')?'larry':'lightx2v');sync();persistSettingsSoon()};
  for(const id of ['h3_no_turbo','h3_latent_upscale','h3_rtx_upscale'])$(id).onchange=()=>{sync();persistSettingsSoon()};
  // Move existing controls so their IDs and existing bindings remain intact.
  const low=$('h3_megapixels').closest('label');for(const node of low.childNodes)if(node.nodeType===3&&node.textContent.trim()){node.textContent='LOWRES SIZE (MP)';break}
  function group(title,ids){const section=document.createElement('section');section.className='h3ControlGroup';const heading=document.createElement('h3');heading.textContent=title;const fields=document.createElement('div');fields.className='fields h3PairedFields';section.append(heading,fields);for(const id of ids)fields.appendChild($(id).closest('label'));panel.appendChild(section)}
  group('Size & output',['h3_megapixels','h3_final_megapixels','h3_aspect_ratio','h3_duration','h3_latent_upscale','h3_rtx_upscale']);
  group('Sampling',['h3_steps','h3_pass1_split','h3_second_pass_sigma','h3_sampler','h3_scheduler','h3_seed','h3_seed_random']);
  group('Turbo',['h3_no_turbo','h3_use_larry','h3_turbo_lora','h3_turbo_strength']);
  group('Latent model',['h3_latent_upscale_model']);
  photos.querySelector('h3').after($('h3_use_multi_image').closest('label'));
  panel.querySelector('fieldset').remove();panel.querySelector('h3').remove();
  generationBody.querySelector(':scope > h3').textContent='Runtime & advanced';
  const original=h3Settings;
  window.h3Settings=()=>{const legacy=original();const settings={...legacy,legacy_inputs:{first_frame_path:legacy.first_frame_path,last_frame_path:legacy.last_frame_path,reference_paths:legacy.reference_paths,reference_subjects:legacy.reference_subjects},workflow_variant:variant(),use_multi_image:chk('h3_use_multi_image'),use_larry:chk('h3_use_larry'),latent_upscale:chk('h3_latent_upscale'),rtx_upscale:chk('h3_rtx_upscale'),final_megapixels:Number(val('h3_final_megapixels')),pass1_split:Number(val('h3_pass1_split')),second_pass_sigma:Number(val('h3_second_pass_sigma')),latent_upscale_model:val('h3_latent_upscale_model'),photo_paths:[1,2,3,4].map(i=>val('h3_photo'+i).trim()),keyframe_positions:[1,2,3,4].map(i=>val('h3_keyframe'+i).trim())};
    if(isNew()){settings.task=settings.workflow_variant==='fflf_20260920'?'fl2v':'r2v';settings.first_frame_path=settings.use_multi_image?settings.photo_paths.find(Boolean)||'':settings.photo_paths[0];settings.last_frame_path=settings.photo_paths[1];settings.reference_paths=settings.photo_paths.filter(Boolean);settings.turbo_enabled=!chk('h3_no_turbo');settings.sampling_preset='custom';settings.pdd_enabled=false;settings.turbo_family=settings.use_larry?'larry':'lightx2v'}
    return settings;
  };
  const apply=applyH3Settings;
  window.applyH3Settings=(h3={},storage={})=>{
    apply({...h3,...(h3.legacy_inputs||{})},storage);setChk('h3_no_turbo',h3.turbo_enabled===false);
    for(const [name,defaultValue] of Object.entries({use_multi_image:false,use_larry:(h3.turbo_family==='larry'||(!h3.turbo_family&&h3.sampler==='h3_turbo')),latent_upscale:true,rtx_upscale:true}))setChk('h3_'+name,h3[name]??defaultValue);
    for(const [name,defaultValue] of Object.entries({final_megapixels:1,pass1_split:Math.min(val('h3_task')==='r2v'?6:3,Math.max(1,Number(h3.steps??6)-1)),second_pass_sigma:2,latent_upscale_model:'minimax_h3_latent_upscaler_3d_bf16.safetensors'}))set('h3_'+name,h3[name]??defaultValue);
    const paths=h3.photo_paths||(h3.task==='r2v'?h3.reference_paths:[h3.first_frame_path,h3.last_frame_path])||[];
    for(let i=1;i<=4;i++){set('h3_photo'+i,paths[i-1]||'');set('h3_keyframe'+i,(h3.keyframe_positions||[])[i-1]||'');void $('h3_photo'+i).refreshPreview()}
    sync();
  };
  window.addEventListener('console-ready',()=>{applyH3Settings(SETTINGS.h3||{},SETTINGS.h3_storage||{});sync()},{once:true});
})();

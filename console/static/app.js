let SETTINGS={}, PROMPTS={}, H3_SUBJECTS={}, H3_SUBJECT_LAST_BLOCK='', H3_COMFYUI_POD=null, JOB_TIMERS={}, PROGRESS_TIMERS={}, H3_LORA_LIBRARY=[], H3_LORA_LIBRARY_LOADED=false, H3_LORA_TRANSFER=null, H3_LORA_TRANSFER_TIMER=null, H3_LORA_XHR=null, MODEL_DOWNLOAD_TIMER=null, MODEL_MANAGER_LOADED=false, CURRENT_BROWSER='', BROWSER_TARGET='', BROWSER_SOURCE='local', BROWSER_STORAGE='h3', BROWSER_SELECTED_ITEM=null, BROWSER_SELECTED_ITEMS={}, BROWSER_VISIBLE_ITEMS=[], BROWSER_LAST_INDEX=-1, RESTORING_PREVIEWS=false, RESTORED_TABS=new Set(), PREVIEW_RESTORE_QUEUE=Promise.resolve(), SAMIMATE_STATE=null, IMAGE_EDITOR={img:null,path:'',target:'',crop:null,lasso:[],points:{positive:[],negative:[]},drag:null,box:null,scale:1,ox:0,oy:0,output:''};
let points={positive:[],negative:[]}, pointFrame={video:null,canvas:null,ctx:null,history:[]};
const $=id=>document.getElementById(id);function log(m){const el=$('log');if(!el)return;const t=new Date().toLocaleTimeString();el.textContent+=`[${t}] ${m}\n`;el.scrollTop=el.scrollHeight}
function setLogHeight(h){h=Math.max(120,Math.min(window.innerHeight-90,Math.round(h)));document.documentElement.style.setProperty('--log-height',h+'px');localStorage.setItem('log_height',String(h));document.body.style.paddingBottom=(document.body.classList.contains('logsHidden')?46:h)+'px'}
function toggleLogs(){document.body.classList.toggle('logsHidden');$('log_toggle_btn').textContent=document.body.classList.contains('logsHidden')?'Show':'Hide';setLogHeight(Number(localStorage.getItem('log_height')||300))}
function clearLogs(){$('log').textContent=''}
function setupLogResize(){let h=$('log_resize_handle'),drag=false;if(!h)return;h.addEventListener('pointerdown',e=>{drag=true;h.setPointerCapture(e.pointerId);document.body.classList.remove('logsHidden');$('log_toggle_btn').textContent='Hide';setLogHeight(window.innerHeight-e.clientY)});h.addEventListener('pointermove',e=>{if(drag)setLogHeight(window.innerHeight-e.clientY)});h.addEventListener('pointerup',e=>{drag=false;try{h.releasePointerCapture(e.pointerId)}catch{}})}
function formatApiError(text){
  try{
    let j=JSON.parse(text),d=j.detail??j;
    if(typeof d==='string')return d;
    if(d&&typeof d==='object'){
      let msg=d.message||JSON.stringify(d);
      if(d.payload_keys)msg+=`\nPayload keys: ${d.payload_keys.join(', ')}`;
      if(d.debug)msg+=`\nDebug: ${JSON.stringify(d.debug)}`;
      return msg;
    }
  }catch{}
  return text;
}
async function api(url,opts={}){const r=await fetch(url,opts);if(!r.ok){let e=new Error(formatApiError(await r.text()));e.status=r.status;throw e}return r.json()}
function val(id){return $(id)?.value ?? ''}function chk(id){return id==='sam_make_masked_video'?true:!!$(id)?.checked}function set(id,v){if($(id))$(id).value=v??''}function setChk(id,v){if($(id))$(id).checked=id==='sam_make_masked_video'?true:!!v}
function intFieldOrDefault(id, fallback){let raw=val(id).trim();if(raw==='')return fallback;let n=parseInt(raw,10);return Number.isFinite(n)?n:fallback}
function setWanMaskPretrimmed(on){set('wan_mask_pretrimmed',on?'1':'')}
function wanMaskPretrimmed(){return val('wan_mask_pretrimmed')==='1'}
function showTab(id){document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));$(id).classList.add('active');document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('active',b.dataset.tab===id));queueTabPreviewRestore(id);if(id==='h3'){void loadH3ModelLibrary(true);if(!H3_LORA_LIBRARY_LOADED){H3_LORA_LIBRARY_LOADED=true;loadH3LoraLibrary(true)}}if(id==='model_manager'&&!MODEL_MANAGER_LOADED){MODEL_MANAGER_LOADED=true;refreshModelManagerFiles(true)}if(id==='image_editor')setTimeout(()=>imageEditorDraw(),0)}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
$('h3_duration')?.addEventListener('input',()=>{updateH3ReferenceVideoFrameCaps();persistSettingsSoon()});
async function init(){const v=await api('/api/version');$('version').textContent=v.version;setupH3ReferenceInputs();setupH3FrameInputs();setupSamimateReferences();setupH3RecentInputs();setupH3Sections();setupDrops();buildVideoTools('inf');buildVideoTools('wan');buildVideoTools('samimate');mountTrimVideoPreview('inf');mountTrimVideoPreview('wan');mountTrimVideoPreview('samimate');buildAudioTools('inf_audio');buildAudioTools('inf_audio2');setupLogResize();setupPoints();setupSamStudio();setupBrowserEvents();setupImageEditor();SETTINGS=await api('/api/settings');applySettings();applyH3Settings(SETTINGS.h3||{},SETTINGS.h3_storage||{});setupSamimateH3Controls();setChk('h3_seed_random',SETTINGS.h3?.seed_random===true||Number(SETTINGS.h3?.seed)<0);void loadH3ModelLibrary(true);void refreshH3ComfyUIStatus(true);setupWorkflowSuite();await loadTtsVoices();await loadPiperStatus();await loadPrompts();await loadH3SubjectLibrary();await loadBrowserFavorites();await loadCacheInfo();await loadPresets();await loadEndpointProfiles();await loadJobHistory();await refreshRecentOutputs();$('inf_input_type').addEventListener('change',updateInfiniteModeUI);$('inf_person_count').addEventListener('change',updateInfiniteModeUI);$('h3_task')?.addEventListener('change',()=>{if(chk('h3_turbo_enabled')&&val('h3_turbo_family')==='lightx2v')selectH3TurboFamily('lightx2v');updateH3TaskUI();updateH3QualityNotice()});$('h3_turbo_enabled')?.addEventListener('change',()=>{set('h3_sampling_preset','custom');updateH3TurboUI(true)});$('h3_sampler')?.addEventListener('change',()=>{set('h3_sampling_preset','custom');updateH3TurboUI(true)});$('h3_scheduler')?.addEventListener('change',()=>{set('h3_sampling_preset','custom');updateH3QualityNotice()});$('h3_cache_enabled')?.addEventListener('change',()=>{set('h3_sampling_preset','custom');updateH3TurboUI(false)});$('h3_seed_random')?.addEventListener('change',updateH3SeedUI);$('h3_lora_file')?.addEventListener('change',h3LoraFileSelected);updateInfiniteModeUI();updateH3TaskUI();updateH3TurboUI(false);updateH3SeedUI();updateH3PreviewVisibility();applyTooltips();setLogHeight(Number(localStorage.getItem('log_height')||300));await browse(SETTINGS.output_dir||'');log('Web UI ready')}
function applySettings(restore=true){
  for(const k of ['infinite_endpoint_id','wan_endpoint_id','timeout_seconds','delivery_mode','s3_endpoint_url','s3_bucket','s3_region','s3_prefix','runpod_volume_root','output_dir','cache_dir','sam_output_dir','cache_max_age_days','hf_cache_dir'])set(k,SETTINGS[k]);
  set('browser_storage',SETTINGS.browser_s3_storage||'h3');BROWSER_STORAGE=val('browser_storage')||'h3';
  let i=SETTINGS.infinitetalk||{},w=SETTINGS.wananimate||{},sm=SETTINGS.samimate||{},sam=SETTINGS.sam||{},tts=SETTINGS.tts||{};
  setChk('cache_cleanup_enabled',SETTINGS.cache_cleanup_enabled!==false);
  set('tts_default_engine',tts.engine||'windows_sapi');set('tts_engine',tts.engine||'windows_sapi');set('inf_tts_engine',tts.engine||'windows_sapi');set('tts_rate',tts.rate??0);set('inf_tts_rate',tts.rate??0);set('tts_volume',tts.volume??100);set('inf_tts_volume',tts.volume??100);set('tts_piper_exe',tts.piper_exe||'');set('tts_piper_model',tts.piper_model||'');set('tts_piper_config',tts.piper_config||'');set('tts_output_path',tts.output_path||'');set('inf_tts_output_path',tts.inf_output_path||'');
  set('inf_input_type',i.input_type);set('inf_person_count',i.person_count);set('inf_image_path',i.image_path||'');set('inf_video_path',i.video_path||'');set('inf_audio_path',i.audio_path||'');set('inf_audio2_path',i.audio2_path||'');set('inf_width',i.width);set('inf_height',i.height);set('inf_max_frame',i.max_frame);setChk('inf_sync_to_audio',i.sync_to_audio);setChk('inf_force_offload',i.force_offload);setChk('inf_network_volume',i.network_volume);set('inf_prompt',i.prompt);set('inf_advanced',i.advanced_json||'{}');set('inf_video_start',i.video_start||'');set('inf_video_end',i.video_end||'');set('inf_video_frame_cap',i.video_frame_cap||'');set('inf_video_crop',i.video_crop||'');set('inf_audio_start',i.audio_start||'');set('inf_audio_end',i.audio_end||'');set('inf_audio2_start',i.audio2_start||'');set('inf_audio2_end',i.audio2_end||'');
  set('wan_mode',w.mode);set('wan_image_path',w.image_path||'');set('wan_video_path',w.video_path||'');set('wan_mask_path',w.mask_path||'');set('wan_masked_video_path',w.masked_video_path||'');setWanMaskPretrimmed(w.mask_pretrimmed);set('wan_width',w.width);set('wan_height',w.height);set('wan_seed',w.seed);set('wan_fps',w.fps);set('wan_cfg',w.cfg);set('wan_steps',w.steps);setChk('wan_pose',w.pose_estimation);setChk('wan_face',w.face_detection);setChk('wan_mask_edit',w.mask_editing);setChk('wan_network_volume',w.network_volume);setChk('wan_control_points',w.control_points_enabled);set('wan_prompt',w.prompt);set('wan_negative_prompt',w.negative_prompt);set('wan_advanced',w.advanced_json||'{}');set('wan_video_start',w.video_start||'');set('wan_video_end',w.video_end||'');set('wan_video_frame_cap',w.video_frame_cap||'');set('wan_video_crop',w.video_crop||'');set('wan_delivery_override',w.delivery_override||'auto');
  set('samimate_video_path',sm.video_path||'');set('samimate_image_path',sm.image_path||'');for(let i=2;i<=9;i++)set(`samimate_reference${i}_path`,(sm.extra_reference_paths||[])[i-2]||'');set('samimate_subject_prompt',sm.subject_prompt||'person');set('samimate_prompt',sm.prompt||'');set('samimate_negative_prompt',sm.negative_prompt||'blurry, low quality, distorted');set('samimate_advanced',sm.advanced_json||'{}');set('samimate_generation_backend',sm.generation_backend||'h3');set('samimate_frame_cap',sm.mask_frame_cap??0);set('samimate_mode',sm.mode||'replace');set('samimate_delivery_override',sm.delivery_override||'auto');set('samimate_width',sm.width||w.width||832);set('samimate_height',sm.height||w.height||480);set('samimate_seed',sm.seed||w.seed||12345);set('samimate_fps',sm.fps||w.fps||16);set('samimate_cfg',sm.cfg||w.cfg||1);set('samimate_steps',sm.steps||w.steps||6);setChk('samimate_pose',sm.pose_estimation!==false);setChk('samimate_face',sm.face_detection!==false);setChk('samimate_mask_edit',true);setChk('samimate_network_volume',sm.network_volume!==false);setChk('samimate_use_wan_points',sm.use_wan_points);set('samimate_video_start',sm.video_start||'');set('samimate_video_end',sm.video_end||'');set('samimate_video_frame_cap',sm.video_frame_cap??'');set('samimate_video_crop',sm.video_crop||'');updateSamimateBackendUI();if($('samimate_h3_controls')?.children.length){samimateSetH3Options(sm.h3_options||{});set('samimate_prompt_mode',sm.prompt_mode||'auto');set('samimate_resolution',sm.resolution||'auto')}
  set('sam_backend',sam.backend||'local_sam3');set('sam3_python',sam.sam3_python||'');set('sam_external_command',sam.external_command||'');set('sam_mask_fill',sam.mask_fill||'black');set('sam_source_path',sam.source_path||'');set('sam_video_start',sam.video_start||'');set('sam_video_end',sam.video_end||'');set('sam_mask_path',sam.mask_path||'');set('sam_mask_video_path',sam.mask_video_path||'');set('sam_overlay_path',sam.overlay_path||'');set('sam_cutout_path',sam.cutout_path||'');set('sam_masked_image_path',sam.masked_image_path||'');set('sam_masked_video_path',sam.masked_video_path||'');setChk('sam_make_masked_video',sam.make_masked_video);set('sam_video_frame_cap',sam.masked_video_frame_cap??180);
  applySecretFieldState();loadCropFields('inf');loadCropFields('wan');loadCropFields('samimate');updateSamBackendNotice();updateInfiniteModeUI();if(restore)setTimeout(()=>queueTabPreviewRestore('inf'),0)
}
function applySecretFieldState(){let configured=SETTINGS.secret_fields_configured||{},fields={'runpod_api_key':'runpod_api_key','s3_access_key_id':'s3_access_key_id','s3_secret_access_key':'s3_secret_access_key','sam.hf_token':'sam_hf_token','h3_storage.s3_access_key_id':'h3_s3_access_key_id','h3_storage.s3_secret_access_key':'h3_s3_secret_access_key','model_download.hf_token':'model_hf_token','model_download.civitai_token':'model_civitai_token'};for(const [path,id] of Object.entries(fields)){let el=$(id);if(!el)continue;el.value='';el.placeholder=configured[path]?'Configured — leave blank to keep':(el.dataset.defaultPlaceholder||el.placeholder||'');}}
function gatherBaseSettings(){
  return {
    runpod_api_key:val('runpod_api_key'),
    infinite_endpoint_id:val('infinite_endpoint_id'),
    wan_endpoint_id:val('wan_endpoint_id'),
    timeout_seconds:+val('timeout_seconds')||3600,
    delivery_mode:val('delivery_mode'),
    browser_s3_storage:val('browser_storage')||'h3',
    s3_endpoint_url:val('s3_endpoint_url'),
    s3_access_key_id:val('s3_access_key_id'),
    s3_secret_access_key:val('s3_secret_access_key'),
    s3_bucket:val('s3_bucket'),
    s3_region:val('s3_region'),
    s3_prefix:val('s3_prefix'),
    runpod_volume_root:val('runpod_volume_root'),
    output_dir:val('output_dir'),
    cache_dir:val('cache_dir'),
    sam_output_dir:val('sam_output_dir'),
    cache_max_age_days:+val('cache_max_age_days')||0,
    cache_cleanup_enabled:chk('cache_cleanup_enabled'),
    hf_cache_dir:val('hf_cache_dir'),
    tts:{engine:val('tts_default_engine')||val('tts_engine')||'windows_sapi',voice:val('tts_voice')||val('inf_tts_voice'),rate:+(val('tts_rate')||0),volume:+(val('tts_volume')||100),piper_exe:val('tts_piper_exe'),piper_model:val('tts_piper_model'),piper_config:val('tts_piper_config'),output_path:val('tts_output_path'),inf_output_path:val('inf_tts_output_path')},
    sam:{backend:val('sam_backend'),sam3_python:val('sam3_python'),hf_token:val('sam_hf_token'),mask_fill:val('sam_mask_fill'),make_masked_video:chk('sam_make_masked_video'),masked_video_frame_cap:intFieldOrDefault('sam_video_frame_cap',180),source_path:val('sam_source_path'),video_start:val('sam_video_start'),video_end:val('sam_video_end'),mask_path:val('sam_mask_path'),mask_video_path:val('sam_mask_video_path'),overlay_path:val('sam_overlay_path'),cutout_path:val('sam_cutout_path'),masked_image_path:val('sam_masked_image_path'),masked_video_path:val('sam_masked_video_path')},
    infinitetalk:{input_type:val('inf_input_type'),person_count:val('inf_person_count'),image_path:val('inf_image_path'),video_path:val('inf_video_path'),audio_path:val('inf_audio_path'),audio2_path:val('inf_audio2_path'),width:+val('inf_width')||512,height:+val('inf_height')||512,max_frame:val('inf_max_frame'),sync_to_audio:chk('inf_sync_to_audio'),force_offload:chk('inf_force_offload'),network_volume:chk('inf_network_volume'),prompt:val('inf_prompt'),advanced_json:val('inf_advanced'),video_start:val('inf_video_start'),video_end:val('inf_video_end'),video_frame_cap:val('inf_video_frame_cap'),video_crop:val('inf_video_crop'),audio_start:val('inf_audio_start'),audio_end:val('inf_audio_end'),audio2_start:val('inf_audio2_start'),audio2_end:val('inf_audio2_end')},
    wananimate:{mode:val('wan_mode'),image_path:val('wan_image_path'),video_path:val('wan_video_path'),mask_path:val('wan_mask_path'),masked_video_path:val('wan_masked_video_path'),mask_pretrimmed:wanMaskPretrimmed(),width:intFieldOrDefault('wan_width',0),height:intFieldOrDefault('wan_height',0),seed:+val('wan_seed')||12345,fps:+val('wan_fps')||16,cfg:+val('wan_cfg')||1,steps:+val('wan_steps')||6,pose_estimation:chk('wan_pose'),face_detection:chk('wan_face'),mask_editing:chk('wan_mask_edit'),network_volume:chk('wan_network_volume'),control_points_enabled:chk('wan_control_points'),prompt:val('wan_prompt'),negative_prompt:val('wan_negative_prompt'),advanced_json:val('wan_advanced'),video_start:val('wan_video_start'),video_end:val('wan_video_end'),video_frame_cap:val('wan_video_frame_cap'),video_crop:val('wan_video_crop'),delivery_override:val('wan_delivery_override')},
    samimate:{h3_options:samimateH3Options(),prompt_mode:val('samimate_prompt_mode'),resolution:val('samimate_resolution'),video_path:val('samimate_video_path'),image_path:val('samimate_image_path'),extra_reference_paths:Array.from({length:8},(_,i)=>val(`samimate_reference${i+2}_path`)),subject_prompt:val('samimate_subject_prompt'),prompt:val('samimate_prompt'),negative_prompt:val('samimate_negative_prompt'),advanced_json:val('samimate_advanced'),generation_backend:val('samimate_generation_backend')||'wan',mask_frame_cap:intFieldOrDefault('samimate_frame_cap',0),mode:val('samimate_mode'),delivery_override:val('samimate_delivery_override'),width:intFieldOrDefault('samimate_width',0),height:intFieldOrDefault('samimate_height',0),seed:+val('samimate_seed')||12345,fps:+val('samimate_fps')||16,cfg:+val('samimate_cfg')||1,steps:+val('samimate_steps')||6,pose_estimation:chk('samimate_pose'),face_detection:chk('samimate_face'),mask_editing:true,network_volume:chk('samimate_network_volume'),video_start:val('samimate_video_start'),video_end:val('samimate_video_end'),video_frame_cap:val('samimate_video_frame_cap'),video_crop:val('samimate_video_crop'),use_wan_points:chk('samimate_use_wan_points')}
  }
}
function setupH3Sections(){document.querySelectorAll('#h3 details.h3Section').forEach(section=>{let key='h3_section_'+section.dataset.h3Section;let saved=localStorage.getItem(key);if(saved!==null)section.open=saved==='open';section.addEventListener('toggle',()=>localStorage.setItem(key,section.open?'open':'closed'))})}
function setupH3RecentInputs(){document.querySelectorAll('#h3 input[id$="_path"]').forEach(input=>{if(!input.dataset.recentBound){input.dataset.recentBound='1';input.addEventListener('focus',()=>showPathRecent(input));input.addEventListener('click',()=>showPathRecent(input));input.addEventListener('input',()=>showPathRecent(input))}let row=input.closest('.inlineField');if(row&&!row.querySelector('.pathRecentTrigger')){let button=document.createElement('button');button.type='button';button.className='secondaryButton pathRecentTrigger';button.textContent='Recent ▾';button.title='Choose a recently used H3 input';button.onclick=event=>togglePathRecent(input.id,event);let clear=[...row.querySelectorAll('button')].find(item=>item.textContent.trim()==='Clear');row.insertBefore(button,clear||null)}})}
function setupH3ReferenceInputs(){let grid=$('h3_reference_grid');if(grid&&!grid.children.length)for(let i=1;i<=9;i++){let card=document.createElement('article');card.className='h3ReferenceCard';card.dataset.picture=String(i);card.innerHTML=`<div class="h3ReferenceTitle"><strong id="h3_picture${i}_label">Reference ${i}</strong><span id="h3_subject${i}_label" class="h3SubjectBadge">Optional subject</span></div><div class="inlineField"><input id="h3_reference${i}_path" placeholder="${i===1?'Image path / HTTPS URL':'Optional image path / HTTPS URL'}"><button type="button" onclick="openNativePicker('h3_reference${i}_path','image')">Browse</button><button type="button" class="secondaryButton" onclick="clearPathInput('h3_reference${i}_path');syncH3SubjectDefinitions()">Clear</button></div><div class="h3ReferencePreview h3PreviewArea"><div class="h3PreviewEmpty" id="h3_reference${i}_empty">No preview</div><img id="h3_reference${i}_preview" class="mediaPreview hidden" alt="H3 picture reference ${i} preview"></div><div class="h3SubjectControls"><label class="h3SubjectEnable"><input id="h3_subject${i}_enabled" type="checkbox"> Use as Subject</label><div class="h3SubjectIdentity"><label>Subject tag<input id="h3_subject${i}_tag" placeholder="Auto, 1, Subject 1, or a name"></label><label>Subject preset<select id="h3_subject${i}_preset"><option value="">Custom definition</option></select></label></div><label>Definition<textarea id="h3_subject${i}_definition" rows="3" placeholder="the woman with short dark hair and a blue jacket"></textarea></label><div class="h3SubjectActions"><button type="button" onclick="applyH3SubjectPreset(${i})">Recall preset</button><button type="button" onclick="saveH3SubjectPreset(${i})">Save to Subject Book</button></div></div>`;grid.appendChild(card);let path=$(`h3_reference${i}_path`),enabled=$(`h3_subject${i}_enabled`),tag=$(`h3_subject${i}_tag`),definition=$(`h3_subject${i}_definition`);path.addEventListener('input',()=>{if(!path.value.trim())clearPreviewForInput(path.id);syncH3SubjectDefinitions();persistSettingsSoon()});path.addEventListener('change',async()=>{let p=path.value.trim();if(p)try{previewForInput(path.id,await previewUrlForPath(p))}catch(e){log(`Reference ${i} preview failed: `+e.message)}syncH3SubjectDefinitions();await persistSettingsQuietly()});for(const el of [enabled,tag,definition])el.addEventListener(el===enabled?'change':'input',()=>{syncH3SubjectDefinitions();persistSettingsSoon()});bindH3ReferenceDrop(card,path,'image')}setupH3MediaReferenceInputs('video',3);setupH3MediaReferenceInputs('audio',3)}
function setupH3MediaReferenceInputs(kind,count){let grid=$(`h3_reference_${kind}_grid`);if(!grid||grid.children.length)return;for(let i=1;i<=count;i++){let stem=`h3_reference_${kind}${i}`,label=kind==='video'?'Video':'Audio',media=kind==='video'?`<video id="${stem}_preview" class="mediaPreview hidden" controls muted></video>`:`<audio id="${stem}_preview" class="mediaPreview hidden" controls></audio>`,extra=kind==='video'?`<div class="h3VideoSettings"><label>Force FPS<input id="${stem}_force_rate" type="number" min="0" max="120" step="1" value="24" title="Set 0 to preserve the source rate."></label><label>Start frame<input id="${stem}_start_frame" type="number" min="0" step="1" value="0"></label><label>Keep every Nth<input id="${stem}_select_every_nth" type="number" min="1" max="1000" step="1" value="1"></label><label>Frame cap (auto)<input id="${stem}_frame_cap" type="number" readonly value="" title="Always matches the H3 output frame length."></label></div><label class="h3SubjectEnable"><input id="${stem}_include_audio" type="checkbox" checked> Include embedded audio as &lt;Video ${i}&gt; audio</label>`:'';let card=document.createElement('article');card.className='h3ReferenceCard';card.innerHTML=`<div class="h3ReferenceTitle"><strong id="${stem}_label">${label} reference ${i}</strong><span class="h3MediaBadge">Optional</span></div><div class="inlineField"><input id="${stem}_path" placeholder="Optional ${kind} path / HTTPS URL"><button type="button" onclick="openNativePicker('${stem}_path','${kind}')">Browse</button><button type="button" class="secondaryButton" onclick="clearPathInput('${stem}_path');updateH3ReferenceLabels()">Clear</button></div><div class="h3ReferencePreview h3PreviewArea"><div class="h3PreviewEmpty" id="${stem}_empty">No preview</div>${media}</div>${extra}`;grid.appendChild(card);let path=$(`${stem}_path`);path.addEventListener('input',()=>{if(!path.value.trim())clearPreviewForInput(path.id);updateH3ReferenceLabels();persistSettingsSoon()});path.addEventListener('change',async()=>{let p=path.value.trim();if(p)try{previewForInput(path.id,await previewUrlForPath(p))}catch(e){log(`${label} reference ${i} preview failed: `+e.message)}updateH3ReferenceLabels();await persistSettingsQuietly()});$(`${stem}_include_audio`)?.addEventListener('change',persistSettingsSoon);for(const suffix of ['force_rate','start_frame','select_every_nth'])$(`${stem}_${suffix}`)?.addEventListener('input',()=>{updateH3ReferenceVideoFrameCaps();persistSettingsSoon()});bindH3ReferenceDrop(card,path,kind)}updateH3ReferenceVideoFrameCaps()}
function bindH3ReferenceDrop(card,path,kind){card.ondragover=e=>{e.preventDefault();card.classList.add('drag')};card.ondragleave=()=>card.classList.remove('drag');card.ondrop=async e=>{e.preventDefault();card.classList.remove('drag');await uploadFileToTarget(e.dataTransfer.files[0],path.id,kind);syncH3SubjectDefinitions();updateH3ReferenceLabels()}}
function h3ReferencePaths(){let out=[];for(let i=1;i<=9;i++){let p=val(`h3_reference${i}_path`).trim();if(p)out.push(p)}return out}
function h3ReferenceVideoPaths(){let out=[];for(let i=1;i<=3;i++){let p=val(`h3_reference_video${i}_path`).trim();if(p)out.push(p)}return out}
function h3ReferenceVideoAudio(){let out=[];for(let i=1;i<=3;i++)if(val(`h3_reference_video${i}_path`).trim())out.push(chk(`h3_reference_video${i}_include_audio`));return out}
function h3OutputFrameCount(){let raw=Math.max(5,Math.round((Number(val('h3_duration'))||2)*24));return raw+(5-(raw%17))%17}
function updateH3ReferenceVideoFrameCaps(){let cap=h3OutputFrameCount();for(let i=1;i<=3;i++)set(`h3_reference_video${i}_frame_cap`,cap)}
function h3ReferenceVideoSettings(){let out=[],cap=h3OutputFrameCount();for(let i=1;i<=3;i++)if(val(`h3_reference_video${i}_path`).trim())out.push({force_rate:Number(val(`h3_reference_video${i}_force_rate`)||0),start_frame:intFieldOrDefault(`h3_reference_video${i}_start_frame`,0),select_every_nth:intFieldOrDefault(`h3_reference_video${i}_select_every_nth`,1),frame_load_cap:cap});return out}
function h3ReferenceAudioPaths(){let out=[];for(let i=1;i<=3;i++){let p=val(`h3_reference_audio${i}_path`).trim();if(p)out.push(p)}return out}
function h3SubjectAssignments(){let out=[];for(let i=1;i<=9;i++)out.push({picture:i,enabled:chk(`h3_subject${i}_enabled`),tag:val(`h3_subject${i}_tag`).trim(),definition:val(`h3_subject${i}_definition`).trim(),preset:val(`h3_subject${i}_preset`)||''});return out}
function applyH3SubjectAssignments(items=[]){let byPicture=new Map((Array.isArray(items)?items:[]).map(x=>[Number(x?.picture),x]));for(let i=1;i<=9;i++){let item=byPicture.get(i)||{};setChk(`h3_subject${i}_enabled`,item.enabled===true);set(`h3_subject${i}_tag`,item.tag||'');set(`h3_subject${i}_definition`,item.definition||'');set(`h3_subject${i}_preset`,item.preset||'')}}
function h3PictureNumberForSlot(slot){let number=0;for(let i=1;i<=slot;i++)if(val(`h3_reference${i}_path`).trim())number++;return val(`h3_reference${slot}_path`).trim()?number:0}
function updateH3PictureLabels(){for(let i=1;i<=9;i++){let number=h3PictureNumberForSlot(i),label=$(`h3_picture${i}_label`);if(label)label.textContent=number?`<Picture ${number}>`:`Reference ${i}`}updateH3ReferenceLabels()}
function updateH3ReferenceLabels(){for(const kind of ['video','audio']){let number=0;for(let i=1;i<=3;i++){let path=val(`h3_reference_${kind}${i}_path`).trim(),label=$(`h3_reference_${kind}${i}_label`);if(path)number++;if(label)label.textContent=path?`<${kind==='video'?'Video':'Audio'} ${number}>`:`${kind==='video'?'Video':'Audio'} reference ${i}`}}}
function h3ExistingSubjectMax(){let text=val('h3_prompt');if(H3_SUBJECT_LAST_BLOCK&&text.startsWith(H3_SUBJECT_LAST_BLOCK)){let keptHeader=H3_SUBJECT_LAST_BLOCK.endsWith('\n')&&!H3_SUBJECT_LAST_BLOCK.endsWith('\n\n');text=(keptHeader?'subject_definitions:\n':'')+text.slice(H3_SUBJECT_LAST_BLOCK.length)}let max=0;for(const match of text.matchAll(/<Subject\s+(\d+)>/gi))max=Math.max(max,Number(match[1])||0);return max}
function h3SubjectTag(raw,fallback){let value=String(raw||'').trim().replace(/^<|>$/g,'').replace(/^Subject\s+/i,'').trim();return `<Subject ${value||fallback}>`}
function h3SubjectDefinitionEntries(){let entries=[],subject=h3ExistingSubjectMax();updateH3PictureLabels();for(let i=1;i<=9;i++){let enabled=chk(`h3_subject${i}_enabled`),definition=val(`h3_subject${i}_definition`).trim(),path=val(`h3_reference${i}_path`).trim(),picture=h3PictureNumberForSlot(i),badge=$(`h3_subject${i}_label`),rawTag=val(`h3_subject${i}_tag`).trim();if(!enabled){if(badge){badge.textContent='Optional subject';badge.classList.remove('active')}continue}let tag=rawTag?h3SubjectTag(rawTag,''):h3SubjectTag('',++subject);if(badge){badge.textContent=tag;badge.classList.add('active')}if(!path||!definition||!picture)continue;let body=definition.replace(/^<Subject\s+[^>]+>\s*(?:is\s*)?/i,'').trim().replace(/<Picture\s+\d+>/gi,`<Picture ${picture}>`).replace(/[.\s]+$/,'');if(!/<Picture\s+\d+>/i.test(body))body+=`, whose appearance comes from <Picture ${picture}>`;entries.push({tag,line:`${tag} is ${body}.`,explicit:!!rawTag})}return entries}
function h3SubjectDefinitionLines(){return h3SubjectDefinitionEntries().map(x=>x.line)}
function h3SubjectDefinitionsBlock(){let lines=h3SubjectDefinitionLines();return lines.length?'subject_definitions:\n'+lines.join('\n')+'\n\n':''}
function escapeH3Regex(value){return String(value).replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}
function syncH3SubjectDefinitions({persist=false}={}){let promptEl=$('h3_prompt');if(!promptEl)return;let body=promptEl.value;if(H3_SUBJECT_LAST_BLOCK&&body.startsWith(H3_SUBJECT_LAST_BLOCK)){let keptHeader=H3_SUBJECT_LAST_BLOCK.endsWith('\n')&&!H3_SUBJECT_LAST_BLOCK.endsWith('\n\n');body=(keptHeader?'subject_definitions:\n':'')+body.slice(H3_SUBJECT_LAST_BLOCK.length)}let entries=h3SubjectDefinitionEntries();for(const entry of entries.filter(x=>x.explicit)){let pattern=new RegExp(`(^|\\n)\\s*${escapeH3Regex(entry.tag)}\\s+is\\s+[^\\n]*(?:\\n\\s*)?`,'gi');body=body.replace(pattern,'$1')}let lines=entries.map(x=>x.line).join('\n');if(lines&&/^subject_definitions:\s*\n/i.test(body)){let headerEnd=body.indexOf('\n')+1;H3_SUBJECT_LAST_BLOCK='subject_definitions:\n'+lines+'\n';promptEl.value=H3_SUBJECT_LAST_BLOCK+body.slice(headerEnd)}else{H3_SUBJECT_LAST_BLOCK=lines?'subject_definitions:\n'+lines+'\n\n':'';promptEl.value=H3_SUBJECT_LAST_BLOCK+body}let enabled=h3SubjectAssignments().filter(x=>x.enabled),ready=entries.length,status=$('h3_subject_prompt_status');if(status)status.textContent=enabled.length?`${ready} subject definition${ready===1?'':'s'} linked${ready<enabled.length?' — add an image and definition to complete the remaining subject.':'. Custom tags replace the matching definition already in the prompt.'}`:'No Subject Book definitions enabled.';if(persist)persistSettingsSoon()}
function updateH3PreviewVisibility(){let hidden=localStorage.getItem('h3_input_previews_hidden')==='true';$('h3')?.classList.toggle('h3PreviewsHidden',hidden);document.querySelectorAll('#h3 .h3PreviewArea').forEach(el=>{if(hidden)el.style.setProperty('display','none','important');else el.style.removeProperty('display')});let button=$('h3_preview_toggle');if(button){button.textContent=hidden?'Show previews':'Hide previews';button.setAttribute('aria-pressed',String(hidden));button.title=hidden?'Show all H3 input previews':'Hide all H3 input previews'}}
function toggleH3InputPreviews(){let hidden=localStorage.getItem('h3_input_previews_hidden')!=='true';localStorage.setItem('h3_input_previews_hidden',String(hidden));updateH3PreviewVisibility()}
async function loadH3SubjectLibrary(){H3_SUBJECTS=await api('/api/h3/subjects');let names=Object.keys(H3_SUBJECTS).sort((a,b)=>a.localeCompare(b));for(const select of document.querySelectorAll('#h3_subject_book_pick,[id^="h3_subject"][id$="_preset"]')){let current=select.value;select.innerHTML='<option value="">'+(select.id==='h3_subject_book_pick'?'Choose a saved subject...':'Custom definition')+'</option>';for(const name of names){let o=document.createElement('option');o.value=name;o.textContent=name;select.appendChild(o)}if(names.includes(current))select.value=current}for(const item of h3SubjectAssignments())if(item.preset&&names.includes(item.preset))set(`h3_subject${item.picture}_preset`,item.preset)}
function applyH3SubjectPreset(i,name=''){name=name||val(`h3_subject${i}_preset`)||val('h3_subject_book_pick');let item=H3_SUBJECTS[name];if(!item)return alert('Choose a subject preset first.');set(`h3_subject${i}_preset`,name);set(`h3_subject${i}_definition`,item.definition||'');setChk(`h3_subject${i}_enabled`,true);let savedImage=String(item.image_path||'').trim();if(savedImage&&!val(`h3_reference${i}_path`).trim()){set(`h3_reference${i}_path`,savedImage);previewUrlForPath(savedImage).then(url=>previewForInput(`h3_reference${i}_path`,url)).catch(e=>log('Subject preset preview failed: '+e.message))}syncH3SubjectDefinitions({persist:true});log(`Recalled Subject Book preset "${name}" on reference slot ${i}`)}
function applyH3SubjectBookToFirstAvailable(){let name=val('h3_subject_book_pick');if(!name)return alert('Choose a subject preset first.');let slot=1;for(let i=1;i<=9;i++)if(val(`h3_reference${i}_path`).trim()&&!chk(`h3_subject${i}_enabled`)){slot=i;break}applyH3SubjectPreset(slot,name)}
async function saveH3SubjectPreset(i){let definition=val(`h3_subject${i}_definition`).trim();if(!definition)return alert('Type a subject definition first.');let name=prompt('Save subject preset as:',val(`h3_subject${i}_preset`)||'');if(!name)return;H3_SUBJECTS=await api('/api/h3/subjects',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,definition,image_path:val(`h3_reference${i}_path`).trim()})});await loadH3SubjectLibrary();set(`h3_subject${i}_preset`,name);set('h3_subject_book_pick',name);log('Saved Subject Book preset: '+name)}
async function deleteH3SubjectPreset(){let name=val('h3_subject_book_pick');if(!name)return alert('Choose a subject preset first.');if(!confirm(`Delete subject preset "${name}"?`))return;H3_SUBJECTS=await api('/api/h3/subjects/'+encodeURIComponent(name),{method:'DELETE'});for(let i=1;i<=9;i++)if(val(`h3_subject${i}_preset`)===name)set(`h3_subject${i}_preset`,'');await loadH3SubjectLibrary();persistSettingsSoon();log('Deleted Subject Book preset: '+name)}
function h3Loras(strict=false){let raw;try{raw=JSON.parse(val('h3_loras')||'[]')}catch(e){if(strict)throw new Error('H3 LoRAs JSON error: '+e.message);return []}if(!Array.isArray(raw)){if(strict)throw new Error('H3 LoRAs must be a JSON array.');return []}let out=[];for(const item of raw){if(!item||typeof item!=='object')continue;let name=String(item.name||'').trim(),strength=Number(item.strength??1);if(!name)continue;if(!Number.isFinite(strength)||strength<0||strength>2){if(strict)throw new Error(`LoRA strength for ${name} must be between 0 and 2.`);strength=Math.max(0,Math.min(2,Number.isFinite(strength)?strength:1))}out.push({name,strength})}return out}
function loraShortName(name){return String(name||'').split(/[\\/]/).pop().replace(/\.safetensors$/i,'')}
function h3Settings(){return {task:val('h3_task')||'fl2v',first_frame_path:val('h3_first_frame_path'),last_frame_path:val('h3_last_frame_path'),reference_paths:h3ReferencePaths(),reference_video_paths:h3ReferenceVideoPaths(),reference_video_audio:h3ReferenceVideoAudio(),reference_video_settings:h3ReferenceVideoSettings(),reference_audio_paths:h3ReferenceAudioPaths(),reference_subjects:h3SubjectAssignments(),subject_generated_prefix:H3_SUBJECT_LAST_BLOCK,use_reference_audio_as_output:chk('h3_use_reference_audio'),prompt:val('h3_prompt'),duration:+val('h3_duration')||2,megapixels:+val('h3_megapixels')||0.2,steps:intFieldOrDefault('h3_steps',chk('h3_turbo_enabled')?6:20),seed:Number(val('h3_seed')||123456789),seed_random:chk('h3_seed_random'),sampling_preset:val('h3_sampling_preset')||'custom',pdd_enabled:val('h3_sampling_preset')==='pdd8',sampler:val('h3_sampler')||'res_multistep',scheduler:val('h3_scheduler')||'simple',cache_enabled:chk('h3_cache_enabled'),cache_threshold:Number(val('h3_cache_threshold')||0.18),attention:val('h3_attention')||'auto',sparse_keep_percent:Number(val('h3_sparse_keep_percent')),sparse_tau:Number(val('h3_sparse_tau')),sparse_start_percent:Number(val('h3_sparse_start_percent')),sparse_end_percent:Number(val('h3_sparse_end_percent')),sparse_trained_weights:chk('h3_sparse_trained_weights'),aspect_ratio:h3AspectRatio(),filename_prefix:val('h3_filename_prefix')||'RunPod_Media_Console_H3',delivery:val('h3_delivery')||'auto',fl2va_model:val('h3_fl2va_model').trim(),ref2va_model:val('h3_ref2va_model').trim(),text_encoder:val('h3_text_encoder').trim(),video_vae:val('h3_video_vae').trim(),audio_vae:val('h3_audio_vae').trim(),turbo_enabled:chk('h3_turbo_enabled'),turbo_lora:val('h3_turbo_lora').trim(),turbo_family:val('h3_turbo_family')||'auto',turbo_strength:Number(val('h3_turbo_strength')||1),loras:h3Loras(),advanced_json:val('h3_advanced')||'{}'}}
function applyH3ReferenceVideoSettings(items=[]){for(let i=1;i<=3;i++){let item=(Array.isArray(items)?items:[])[i-1]||{};set(`h3_reference_video${i}_force_rate`,item.force_rate??24);set(`h3_reference_video${i}_start_frame`,item.start_frame??item.skip_first_frames??0);set(`h3_reference_video${i}_select_every_nth`,item.select_every_nth??1)}updateH3ReferenceVideoFrameCaps()}
function applyH3Settings(h3={},storage={}){set('h3_endpoint_id',SETTINGS.h3_endpoint_id||'s1e3i9book2zhh');set('h3_task',h3.task||'fl2v');set('h3_first_frame_path',h3.first_frame_path||'');set('h3_last_frame_path',h3.last_frame_path||'');for(let i=1;i<=9;i++)set(`h3_reference${i}_path`,(h3.reference_paths||[])[i-1]||'');for(let i=1;i<=3;i++){set(`h3_reference_video${i}_path`,(h3.reference_video_paths||[])[i-1]||'');setChk(`h3_reference_video${i}_include_audio`,(h3.reference_video_audio||[])[i-1]!==false);set(`h3_reference_audio${i}_path`,(h3.reference_audio_paths||[])[i-1]||(i===1?(h3.audio_path||''):''))}applyH3ReferenceVideoSettings(h3.reference_video_settings||[]);applyH3SubjectAssignments(h3.reference_subjects||[]);setChk('h3_use_reference_audio',h3.use_reference_audio_as_output);set('h3_prompt',h3.prompt||'A cinematic shot with natural, stable motion.');set('h3_duration',h3.duration??2);set('h3_megapixels',h3.megapixels??0.2);set('h3_steps',h3.steps??6);set('h3_seed',h3.seed??123456789);let legacyTurbo=h3.turbo_enabled!==false,preset=h3.sampling_preset||(legacyTurbo?'stock_turbo':'custom');set('h3_sampling_preset',preset);set('h3_sampler',h3.sampler||(legacyTurbo?'h3_turbo':'res_multistep'));set('h3_scheduler',h3.scheduler||(legacyTurbo?'simple':'beta'));setChk('h3_cache_enabled',h3.cache_enabled===true);set('h3_cache_threshold',h3.cache_threshold??0.18);set('h3_attention',h3.attention||'auto');for(const [key,def] of Object.entries({sparse_keep_percent:10,sparse_tau:1.3,sparse_start_percent:0.2,sparse_end_percent:1}))set('h3_'+key,h3[key]??def);setChk('h3_sparse_trained_weights',h3.sparse_trained_weights===true);setH3AspectRatio(h3.aspect_ratio||'16:9 (Widescreen)');set('h3_filename_prefix',h3.filename_prefix||'RunPod_Media_Console_H3');set('h3_delivery',h3.delivery==='s3_url'?'volume_path':(h3.delivery||'auto'));set('h3_fl2va_model',h3.fl2va_model||'minimax_h3_fl2va_pruned_int8_convrot.safetensors');set('h3_ref2va_model',h3.ref2va_model||'minimax_h3_ref2va_pruned_int8_convrot.safetensors');set('h3_text_encoder',h3.text_encoder||'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors');set('h3_video_vae',h3.video_vae||'minimax_h3_video_vae_fp16.safetensors');set('h3_audio_vae',h3.audio_vae||'minimax_h3_audio_vae_fp32.safetensors');setChk('h3_turbo_enabled',legacyTurbo);set('h3_turbo_lora',h3.turbo_lora||'minimax_h3_turbo_v4_step600_ema.safetensors');set('h3_turbo_family',h3.turbo_family||'auto');set('h3_turbo_strength',h3.turbo_strength??1);set('h3_loras',JSON.stringify(Array.isArray(h3.loras)?h3.loras:[],null,2));set('h3_advanced',h3.advanced_json||'{}');set('h3_network_volume_id',storage.network_volume_id||'vgc3ky6r6y');set('h3_s3_endpoint_url',storage.s3_endpoint_url||'');set('h3_s3_access_key_id',storage.s3_access_key_id||'');set('h3_s3_secret_access_key',storage.s3_secret_access_key||'');set('h3_s3_bucket',storage.s3_bucket||storage.network_volume_id||'vgc3ky6r6y');set('h3_s3_region',storage.s3_region||'eu-ro-1');set('h3_s3_prefix',storage.s3_prefix||'minimax-h3');set('h3_runpod_volume_root',storage.runpod_volume_root||'/runpod-volume');H3_SUBJECT_LAST_BLOCK=h3.subject_generated_prefix||'';syncH3SubjectDefinitions();updateH3ReferenceVideoFrameCaps();updateH3TaskUI();updateH3TurboUI(false);renderH3LoraActive()}
function gatherSettings(){let s=gatherBaseSettings();return {...s,h3_endpoint_id:val('h3_endpoint_id')||'s1e3i9book2zhh',h3:h3Settings(),h3_storage:{network_volume_id:val('h3_network_volume_id')||'vgc3ky6r6y',s3_endpoint_url:val('h3_s3_endpoint_url'),s3_access_key_id:val('h3_s3_access_key_id'),s3_secret_access_key:val('h3_s3_secret_access_key'),s3_bucket:val('h3_s3_bucket')||val('h3_network_volume_id')||'vgc3ky6r6y',s3_region:val('h3_s3_region')||'eu-ro-1',s3_prefix:val('h3_s3_prefix')||'minimax-h3',runpod_volume_root:val('h3_runpod_volume_root')||'/runpod-volume'},model_download:{hf_token:val('model_hf_token'),civitai_token:val('model_civitai_token')}}}
function updateH3TaskUI(){let r2v=val('h3_task')==='r2v';$('h3_fl2v_inputs')?.classList.toggle('hidden',val('h3_task')!=='fl2v');$('h3_r2v_inputs')?.classList.toggle('hidden',!r2v)}
function selectH3ModelMatch(id,needle,fallback=''){let select=$(id);if(!select)return;let option=[...select.options].find(o=>o.value.toLowerCase().includes(String(needle).toLowerCase()));if(option)select.value=option.value;else if(fallback)set(id,fallback)}
function applyH3SamplingPreset(name){if(name==='pdd8'){let kind=val('h3_task')==='r2v'?'Ref2VA':'FL2VA',match=H3_LORA_LIBRARY.find(x=>x.name.includes(kind+'-Acc-8Step')&&x.name.includes('comfy'));if(!match){set('h3_sampling_preset','custom');set('h3_lora_remote_url',`https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/f4cac997f880e93cf6940af61ee8d58ef31ff7f3/loras/MiniMax-H3-${kind}-Acc-8Step_comfy.safetensors`);alert('Import the prepared PDD LoRA URL in the LoRA library, then select this preset again.');return}selectH3ModelMatch(val('h3_task')==='r2v'?'h3_ref2va_model':'h3_fl2va_model',val('h3_task')==='r2v'?'minimax_h3_ref2va_pruned':'minimax_h3_fl2va_pruned',val('h3_task')==='r2v'?'minimax_h3_ref2va_pruned_int8_convrot.safetensors':'minimax_h3_fl2va_pruned_int8_convrot.safetensors');let items=h3Loras().filter(x=>!x.name.includes('Acc-8Step'));items.push({name:match.name,strength:1});setH3Loras(items);setChk('h3_turbo_enabled',false);set('h3_sampler','euler');set('h3_scheduler','simple');set('h3_steps',8);setChk('h3_cache_enabled',false);set('h3_attention','native')}else if(name==='stock_turbo'){selectH3ModelMatch(val('h3_task')==='r2v'?'h3_ref2va_model':'h3_fl2va_model',val('h3_task')==='r2v'?'minimax_h3_ref2va_pruned':'minimax_h3_fl2va_pruned',val('h3_task')==='r2v'?'minimax_h3_ref2va_pruned_int8_convrot.safetensors':'minimax_h3_fl2va_pruned_int8_convrot.safetensors');setChk('h3_turbo_enabled',true);set('h3_turbo_family','larry');set('h3_turbo_lora','minimax_h3_turbo_v4_step600_ema.safetensors');set('h3_sampler','h3_turbo');set('h3_scheduler','simple');set('h3_steps',6);setChk('h3_cache_enabled',false);set('h3_attention','auto')}else if(name==='10eros_multires'||name==='10eros_er_sde'){if(!/10eros/i.test(val(val('h3_task')==='r2v'?'h3_ref2va_model':'h3_fl2va_model'))){log('Select the exact 10Eros model version first; no checkpoint was substituted.');updateH3QualityNotice();return}setChk('h3_turbo_enabled',false);set('h3_sampler',name==='10eros_er_sde'?'er_sde':'res_multistep');set('h3_scheduler','simple');set('h3_steps',6);setChk('h3_cache_enabled',false);set('h3_attention','auto')}updateH3TurboUI(false);persistSettingsSoon()}
function applyH3CleanBaseline(){let preset=val('h3_sampling_preset');if(preset==='custom'){log('Choose a sampling preset first; custom model and sampling settings were preserved.');return}applyH3SamplingPreset(preset);updateH3QualityNotice();persistSettingsQuietly();let count=h3Loras().length;log(`Applied H3 sampling preset without changing ${count} active workflow LoRA${count===1?'':'s'}`)}
function updateH3QualityNotice(){let el=$('h3_quality_notice');if(!el)return;let issues=[],preset=val('h3_sampling_preset'),model=val(val('h3_task')==='r2v'?'h3_ref2va_model':'h3_fl2va_model');if(/10eros/i.test(model)&&/beta[_-]?3/i.test(model))issues.push('MODEL WARNING: the author reports beta3 is corrupted for image/reference starts; sampling presets cannot repair this checkpoint. Choose a compatible stock model or a newer corrected checkpoint');if(chk('h3_cache_enabled'))issues.push('FirstBlockCache is on and can reduce reference accuracy');if(Number(val('h3_megapixels'))>0.98)issues.push('Resolution is above the official 0.98 MP example; check the resulting dimensions against model limits');if(preset.startsWith('10eros')&&!/10eros/i.test(model))issues.push('10Eros preset is selected but the active model is not 10Eros');let count=h3Loras().length;el.textContent=issues.length?'Quality check: '+issues.join('. ')+'.':`Configuration: attention ${val('h3_attention')}, cache ${chk('h3_cache_enabled')?'on':'off'}, separate Turbo LoRA ${chk('h3_turbo_enabled')?'on':'off'}, ${count} workflow LoRAs preserved. Generation quality has not been verified.`;el.classList.toggle('good',!issues.length)}
function renderH3ComfyUIStatus(result){H3_COMFYUI_POD=result?.pod||null;let status=$('h3_comfyui_status'),launch=$('h3_comfyui_launch'),open=$('h3_comfyui_open'),stop=$('h3_comfyui_stop'),pod=H3_COMFYUI_POD;if(!pod){if(status)status.textContent=`No Pod found on volume ${result?.network_volume_id||'?'}`;if(launch)launch.textContent='Create H3 ComfyUI';if(open)open.disabled=true;if(stop)stop.disabled=true;return}let cost=Number(pod.cost_per_hour),costText=Number.isFinite(cost)?` · $${cost.toFixed(2)}/hr`:'';if(status)status.textContent=`${pod.name} · ${pod.status}${costText}`;if(launch){launch.textContent=pod.status==='RUNNING'?'Running':'Start H3 ComfyUI';launch.disabled=pod.status==='RUNNING'}if(open)open.disabled=pod.status!=='RUNNING';if(stop)stop.disabled=pod.status!=='RUNNING'}
async function refreshH3ComfyUIStatus(quiet=false){try{let result=await api('/api/h3/comfyui/status');renderH3ComfyUIStatus(result);if(!quiet)log('H3 ComfyUI Pod status refreshed')}catch(e){H3_COMFYUI_POD=null;if($('h3_comfyui_status'))$('h3_comfyui_status').textContent='Pod status unavailable';if(!quiet)log('H3 ComfyUI status error: '+e.message)}}
async function launchH3ComfyUI(){let pod=H3_COMFYUI_POD,cost=Number(pod?.cost_per_hour),action=pod?'start the existing stopped Pod':'create a new Pod',costText=Number.isFinite(cost)?` The listed GPU rate is $${cost.toFixed(2)}/hour.`:' This starts billable GPU time.';if(!confirm(`This will ${action} with the H3 network volume mounted at /workspace.${costText}\n\nThe app will also add the H3 model paths to ComfyUI extra_model_paths.yaml while preserving existing entries. Continue?`))return;let button=$('h3_comfyui_launch');try{button.disabled=true;button.textContent='Launching…';let result=await api('/api/h3/comfyui/launch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm_cost:true,pod_id:pod?.id||'',gpu_type:val('h3_comfyui_gpu')})});renderH3ComfyUIStatus({pod:result.pod});log(`${result.created?'Created':'Started'} H3 ComfyUI Pod ${result.pod?.id||''}; models mapped from /workspace/models`);setTimeout(()=>refreshH3ComfyUIStatus(true),12000)}catch(e){log('H3 ComfyUI launch error: '+e.message);alert('H3 ComfyUI launch error: '+e.message);await refreshH3ComfyUIStatus(true)}finally{if(button&&!H3_COMFYUI_POD?.ready)button.disabled=false}}
function openH3ComfyUI(){if(!H3_COMFYUI_POD?.url)return alert('Start the H3 ComfyUI Pod first.');window.open(H3_COMFYUI_POD.url,'_blank','noopener')}
async function stopH3ComfyUI(){let pod=H3_COMFYUI_POD;if(!pod?.id)return;if(!confirm(`Stop ${pod.name}? GPU billing will stop; the H3 network-volume files remain.`))return;let button=$('h3_comfyui_stop');try{button.disabled=true;button.textContent='Stopping…';let result=await api('/api/h3/comfyui/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm:true,pod_id:pod.id})});renderH3ComfyUIStatus({pod:result.pod});log(`Stopped H3 ComfyUI Pod ${pod.id}`)}catch(e){log('H3 ComfyUI stop error: '+e.message);alert('H3 ComfyUI stop error: '+e.message);await refreshH3ComfyUIStatus(true)}finally{button.textContent='Stop'}}
function updateH3TurboUI(adjustSteps=false){let enabled=chk('h3_turbo_enabled'),steps=$('h3_steps'),notice=$('h3_turbo_notice');if(steps){steps.removeAttribute('max');steps.min='1'}for(const id of ['h3_turbo_lora','h3_turbo_family','h3_turbo_strength'])if($(id))$(id).disabled=!enabled;if($('h3_cache_threshold'))$('h3_cache_threshold').disabled=!chk('h3_cache_enabled');if(notice)notice.textContent=enabled?'Turbo training counts are recommendations. Your step count is preserved; additional steps may change quality and take longer. LightX2V uses its task-matched LoRA and native flow shifts.':'Steps accept any positive whole number. PDD recommends 8; other presets are starting points.';updateH3QualityNotice()}
function updateH3SeedUI(){if($('h3_seed'))$('h3_seed').disabled=chk('h3_seed_random')}
function h3AspectRatio(){return val('h3_aspect_ratio')==='custom'?(val('h3_aspect_ratio_custom').trim()||'16:9 (Widescreen)'):(val('h3_aspect_ratio')||'16:9 (Widescreen)')}
function setH3AspectRatio(value){let ratio=String(value||'16:9 (Widescreen)'),select=$('h3_aspect_ratio');if(!select)return;let known=[...select.options].some(o=>o.value===ratio&&o.value!=='custom');select.value=known?ratio:'custom';set('h3_aspect_ratio_custom',known?'':ratio);updateH3AspectRatioUI()}
function updateH3AspectRatioUI(){$('h3_aspect_ratio_custom')?.classList.toggle('hidden',val('h3_aspect_ratio')!=='custom')}
function setH3LoraStatus(message,busy=false){let el=$('h3_lora_status');if(el)el.textContent=message||'';for(const id of ['h3_lora_upload_button','h3_lora_import_button'])if($(id))$(id).disabled=busy}
function h3LoraFileSelected(){let file=$('h3_lora_file')?.files?.[0];if(!file)return;let size=file.size>=1024*1024*1024?`${(file.size/(1024**3)).toFixed(2)} GB`:`${(file.size/(1024**2)).toFixed(1)} MB`;setH3LoraStatus(`Ready to upload ${file.name} (${size}). Click Upload local LoRA.`);log(`Selected H3 LoRA: ${file.name} (${size})`)}
function renderH3LoraTransfer(){if(!H3_LORA_TRANSFER)return;let elapsed=(Date.now()-H3_LORA_TRANSFER.started)/1000,progress=$('h3_lora_progress'),pct=H3_LORA_TRANSFER.percent;if(progress){progress.classList.remove('hidden');if(Number.isFinite(pct)){progress.value=pct}else{progress.removeAttribute('value')}}let phase=H3_LORA_TRANSFER.phase||H3_LORA_TRANSFER.label,setPct=Number.isFinite(pct)?` ${Math.round(pct)}%`:'';setH3LoraStatus(`${phase}${setPct} | elapsed ${fmtDurationSec(elapsed)}`,true)}
function beginH3LoraTransfer(label){clearInterval(H3_LORA_TRANSFER_TIMER);H3_LORA_TRANSFER={label,phase:label,percent:null,started:Date.now()};renderH3LoraTransfer();H3_LORA_TRANSFER_TIMER=setInterval(renderH3LoraTransfer,1000)}
function updateH3LoraTransfer(phase,percent=null){if(!H3_LORA_TRANSFER)return;H3_LORA_TRANSFER.phase=phase;H3_LORA_TRANSFER.percent=Number.isFinite(percent)?percent:null;renderH3LoraTransfer()}
function endH3LoraTransfer(){clearInterval(H3_LORA_TRANSFER_TIMER);H3_LORA_TRANSFER_TIMER=null;H3_LORA_TRANSFER=null;H3_LORA_XHR=null;let progress=$('h3_lora_progress');if(progress){progress.classList.add('hidden');progress.value=0}for(const id of ['h3_lora_upload_button','h3_lora_import_button'])if($(id))$(id).disabled=false}
function uploadH3LoraForm(form){return new Promise((resolve,reject)=>{let xhr=new XMLHttpRequest();H3_LORA_XHR=xhr;xhr.open('POST','/api/h3/loras/upload');xhr.upload.onprogress=e=>{if(e.lengthComputable)updateH3LoraTransfer('Transferring LoRA into the app',Math.min(99,(e.loaded/e.total)*100))};xhr.upload.onload=()=>updateH3LoraTransfer('Local transfer complete; copying LoRA to the H3 volume',null);xhr.onerror=()=>{let e=new Error('Local LoRA transfer failed');e.status=xhr.status||0;reject(e)};xhr.onload=()=>{let body=xhr.responseText||'';if(xhr.status>=200&&xhr.status<300){try{resolve(JSON.parse(body))}catch{reject(new Error('LoRA upload returned invalid JSON'))}}else{let e=new Error(formatApiError(body)||`LoRA upload failed with HTTP ${xhr.status}`);e.status=xhr.status;reject(e)}};xhr.send(form)})}
function setH3Loras(items,persist=true){set('h3_loras',JSON.stringify(Array.isArray(items)?items:[],null,2));renderH3LoraActive();if(persist)persistSettingsQuietly()}
function renderH3LoraActive(){let el=$('h3_lora_active');if(!el)return;let items=h3Loras();if(!items.length)el.innerHTML='<span class="loraEmpty">No active H3 LoRAs</span>';else el.innerHTML=items.map((item,i)=>`<div class="h3LoraActiveRow"><span title="${escapeHtml(item.name)}"><strong>${escapeHtml(loraShortName(item.name))}</strong><small>${escapeHtml(item.name)}</small></span><label>Strength<input type="number" min="0" max="2" step="0.05" value="${escapeHtml(item.strength)}" onchange="updateH3LoraStrength(${i},this.value)"></label><button type="button" onclick="removeH3LoraFromWorkflow(${i})">Remove</button></div>`).join('');updateH3QualityNotice()}
function updateH3LoraStrength(index,value){let items=h3Loras(),strength=Number(value);if(!Number.isFinite(strength)||strength<0||strength>2)return alert('H3 LoRA strength must be between 0 and 2.');if(!items[index])return;items[index].strength=strength;setH3Loras(items)}
function removeH3LoraFromWorkflow(index){let items=h3Loras(),removed=items[index];items.splice(index,1);setH3Loras(items);if(removed)log(`Removed H3 LoRA from workflow: ${removed.name}`)}
function renderH3LoraLibrary(selected=''){let sel=$('h3_lora_library_select');if(!sel)return;let current=selected||sel.value;sel.innerHTML='<option value="">Choose a LoRA from the H3 volume...</option>'+H3_LORA_LIBRARY.map(item=>`<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}${item.size_mb?` (${escapeHtml(item.size_mb)} MB)`:''}</option>`).join('');if(current&&H3_LORA_LIBRARY.some(item=>item.name===current))sel.value=current}
async function loadH3LoraLibrary(quiet=false){if(!H3_LORA_TRANSFER)setH3LoraStatus('Loading H3 LoRA library...');try{let r=await api('/api/h3/loras');H3_LORA_LIBRARY=Array.isArray(r.items)?r.items:[];renderH3LoraLibrary();if(!H3_LORA_TRANSFER)setH3LoraStatus(`${H3_LORA_LIBRARY.length} LoRA${H3_LORA_LIBRARY.length===1?'':'s'} available in /runpod-volume/${r.prefix}`);if(!quiet)log(`Loaded ${H3_LORA_LIBRARY.length} H3 LoRA files`)}catch(e){H3_LORA_LIBRARY=[];renderH3LoraLibrary();if(!H3_LORA_TRANSFER)setH3LoraStatus('LoRA library unavailable: '+e.message);if(!quiet){log('H3 LoRA library error: '+e.message);alert('H3 LoRA library error: '+e.message)}}}
function fillH3ModelSelect(id,names=[]){let select=$(id);if(!select)return;let current=select.value,unique=[...new Set((names||[]).filter(Boolean))].sort((a,b)=>a.localeCompare(b));select.replaceChildren();if(current&&!unique.includes(current))unique.unshift(current);for(const name of unique){let option=document.createElement('option');option.value=name;option.textContent=name+(name===current&&!names.includes(name)?' (saved; not listed)':'');select.appendChild(option)}if(current)select.value=current}
async function loadH3ModelLibrary(quiet=false){let status=$('h3_model_status');if(status)status.textContent='Loading models from the H3 volume...';try{let r=await api('/api/h3/models'),c=r.categories||{},allDiffusion=c.diffusion_models||[],fl2va=[...new Set([...(c.fl2va||[]),...allDiffusion])],ref2va=[...new Set([...(c.ref2va||[]),...allDiffusion])];fillH3ModelSelect('h3_fl2va_model',fl2va);fillH3ModelSelect('h3_ref2va_model',ref2va);fillH3ModelSelect('h3_text_encoder',c.text_encoders);fillH3ModelSelect('h3_video_vae',c.video_vaes||c.vaes);fillH3ModelSelect('h3_audio_vae',c.audio_vaes||c.vaes);fillH3ModelSelect('h3_turbo_lora',c.turbo_loras);samimateRefreshModelOptions();let total=allDiffusion.length+(c.text_encoders||[]).length+(c.vaes||[]).length;if(status)status.textContent=`${total} model files loaded from ${r.bucket}; ${(c.turbo_loras||[]).length} Turbo LoRA option(s).`;if(!quiet)log('Refreshed H3 model dropdowns from the network volume')}catch(e){if(status)status.textContent='Model library unavailable: '+e.message;if(!quiet){log('H3 model library error: '+e.message);alert('H3 model library error: '+e.message)}}}
function modelDownloadByteText(bytes){let n=Number(bytes||0);if(n>=1024**3)return`${(n/1024**3).toFixed(2)} GB`;if(n>=1024**2)return`${(n/1024**2).toFixed(1)} MB`;if(n>=1024)return`${(n/1024).toFixed(1)} KB`;return`${n} B`}
function renderModelManagerFiles(result){let el=$('model_manager_files'),items=result.items||[];if($('model_manager_destination'))$('model_manager_destination').textContent=`${result.bucket}: /runpod-volume/${result.destination} — ${items.length} file${items.length===1?'':'s'}`;if(!el)return;if(!items.length){el.innerHTML='<span class="loraEmpty">No model files in this destination.</span>';return}el.innerHTML=items.map(item=>`<div class="modelManagerFile"><span title="${escapeHtml(item.volume_path)}"><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.key)}</small></span><span>${escapeHtml(item.size_mb)} MB</span></div>`).join('')}
async function refreshModelManagerFiles(quiet=true){let el=$('model_manager_files');if(el)el.innerHTML='<span class="loraEmpty">Loading H3 volume inventory...</span>';try{let category=val('model_download_category')||'diffusion_models',r=await api('/api/h3/model-manager/files?category='+encodeURIComponent(category));renderModelManagerFiles(r);if(!quiet)log(`Loaded ${r.items.length} ${category} file(s) from the H3 volume`)}catch(e){if(el)el.innerHTML=`<span class="loraEmpty">${escapeHtml(e.message)}</span>`;if(!quiet)log('H3 model inventory error: '+e.message)}}
function renderModelDownloadJob(job){let status=$('model_download_status'),progress=$('model_download_progress'),details=$('model_download_details'),button=$('model_download_button'),pct=Number(job.percent),terminal=['COMPLETED','FAILED','EXISTS'].includes(job.status);if(status)status.textContent=job.phase||job.status;if(progress){progress.classList.remove('hidden');if(Number.isFinite(pct)){progress.value=pct}else progress.removeAttribute('value')}let transferred=modelDownloadByteText(job.downloaded_bytes),total=job.total_bytes?` / ${modelDownloadByteText(job.total_bytes)}`:'';if(details)details.textContent=[`Status: ${job.status}`,job.filename?`File: ${job.filename}`:'',job.key?`Key: ${job.key}`:'',job.downloaded_bytes?`Transferred: ${transferred}${total}`:'',job.volume_path?`Volume: ${job.volume_path}`:'',job.error?`Error: ${job.error}`:''].filter(Boolean).join('\n');if(button)button.disabled=!terminal&&job.status!=='FAILED';return terminal}
async function pollModelDownload(jobId){clearInterval(MODEL_DOWNLOAD_TIMER);let finished=false,poll=async()=>{try{let job=await api('/api/h3/model-manager/import/'+encodeURIComponent(jobId)),done=renderModelDownloadJob(job);if(done){finished=true;clearInterval(MODEL_DOWNLOAD_TIMER);MODEL_DOWNLOAD_TIMER=null;$('model_download_button').disabled=false;if(job.status==='COMPLETED'){log(`H3 model import completed: ${job.volume_path}`);await refreshModelManagerFiles(true);await loadH3ModelLibrary(true);if((val('model_download_category')||'')==='loras')await loadH3LoraLibrary(true)}else log(`H3 model import ${job.status.toLowerCase()}: ${job.error||job.phase}`)}}catch(e){finished=true;clearInterval(MODEL_DOWNLOAD_TIMER);MODEL_DOWNLOAD_TIMER=null;$('model_download_button').disabled=false;if($('model_download_status'))$('model_download_status').textContent='Status error: '+e.message}};await poll();if(!finished)MODEL_DOWNLOAD_TIMER=setInterval(poll,1200)}
async function startModelDownload(){let url=val('model_download_url').trim(),name=val('model_download_name').trim(),category=val('model_download_category')||'diffusion_models',overwrite=chk('model_download_overwrite');if(!url)return alert('Enter a Hugging Face, Civitai, or direct HTTPS model URL.');if(overwrite&&!confirm('Replace an existing file with the same destination name?'))return;let button=$('model_download_button');try{button.disabled=true;if($('model_download_status'))$('model_download_status').textContent='Saving provider settings...';await saveSettings();let result=await api('/api/h3/model-manager/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url,name,category,overwrite})});if($('model_download_status'))$('model_download_status').textContent='Download queued';log(`Queued H3 ${category} download: ${url}`);await pollModelDownload(result.job_id)}catch(e){button.disabled=false;if($('model_download_status'))$('model_download_status').textContent='Download could not start';log('H3 model download error: '+e.message);alert('H3 model download error: '+e.message)}}
function addH3LoraToWorkflow(name=''){name=String(name||val('h3_lora_library_select')).trim();if(!name)return alert('Choose an H3 LoRA first.');let strength=Number(val('h3_lora_strength')||1);if(!Number.isFinite(strength)||strength<0||strength>2)return alert('H3 LoRA strength must be between 0 and 2.');let items=h3Loras(),existing=items.find(item=>item.name===name);if(existing)existing.strength=strength;else items.push({name,strength});setH3Loras(items);log(`${existing?'Updated':'Added'} H3 LoRA: ${name} at ${strength}`)}
async function uploadH3Lora(overwrite=false){let input=$('h3_lora_file'),file=input?.files?.[0];if(!file)return alert('Choose a local .safetensors file first.');if(!file.name.toLowerCase().endsWith('.safetensors'))return alert('H3 LoRAs must be .safetensors files.');let retryOverwrite=false;try{beginH3LoraTransfer(`Preparing ${file.name}`);await saveSettings();let form=new FormData();form.append('file',file,file.name);form.append('overwrite',String(overwrite));let r=await uploadH3LoraForm(form);updateH3LoraTransfer('Refreshing the H3 LoRA library',100);await loadH3LoraLibrary(true);renderH3LoraLibrary(r.item.name);addH3LoraToWorkflow(r.item.name);endH3LoraTransfer();setH3LoraStatus(`${r.overwritten?'Replaced':'Uploaded'} ${r.item.name} (${r.item.size_mb} MB). It is active in this workflow.`);input.value='';log(`H3 LoRA ${r.overwritten?'replaced':'uploaded'} and activated: ${r.item.volume_path}`)}catch(e){if(e.status===409&&!overwrite&&confirm(`${e.message}. Replace the existing file?`))retryOverwrite=true;else{endH3LoraTransfer();setH3LoraStatus('Local LoRA upload failed: '+e.message);log('H3 LoRA upload error: '+e.message);alert('H3 LoRA upload error: '+e.message)}}finally{if(H3_LORA_TRANSFER)endH3LoraTransfer()}if(retryOverwrite)return uploadH3Lora(true)}
async function importH3Lora(overwrite=false){let url=val('h3_lora_remote_url').trim(),name=val('h3_lora_remote_name').trim();if(!url)return alert('Enter a remote LoRA download URL first.');if(name&&!name.toLowerCase().endsWith('.safetensors'))return alert('The remote LoRA filename must end with .safetensors.');let retryOverwrite=false;try{beginH3LoraTransfer('Downloading remote LoRA and copying it to the H3 volume');await saveSettings();let r=await api('/api/h3/loras/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url,name,overwrite})});updateH3LoraTransfer('Refreshing the H3 LoRA library',100);await loadH3LoraLibrary(true);renderH3LoraLibrary(r.item.name);addH3LoraToWorkflow(r.item.name);endH3LoraTransfer();setH3LoraStatus(`${r.overwritten?'Replaced':'Imported'} ${r.item.name} (${r.item.size_mb} MB). It is active in this workflow.`);set('h3_lora_remote_url','');set('h3_lora_remote_name','');log(`H3 remote LoRA ${r.overwritten?'replaced':'imported'} and activated: ${r.item.volume_path}`)}catch(e){if(e.status===409&&!overwrite&&confirm(`${e.message}. Replace the existing file?`))retryOverwrite=true;else{endH3LoraTransfer();setH3LoraStatus('Remote LoRA import failed: '+e.message);log('H3 remote LoRA error: '+e.message);alert('H3 remote LoRA error: '+e.message)}}finally{if(H3_LORA_TRANSFER)endH3LoraTransfer()}if(retryOverwrite)return importH3Lora(true)}
function payloadSettings(){let s=gatherSettings();return {...s,tts:{engine:s.tts.engine,voice:s.tts.voice,rate:s.tts.rate,volume:s.tts.volume,piper_exe:s.tts.piper_exe,piper_model:s.tts.piper_model,piper_config:s.tts.piper_config},sam:{backend:s.sam.backend,sam3_python:s.sam.sam3_python,hf_token:s.sam.hf_token,mask_fill:s.sam.mask_fill,make_masked_video:s.sam.make_masked_video,masked_video_frame_cap:s.sam.masked_video_frame_cap},infinitetalk:{},wananimate:{},samimate:{}}}
async function saveSettings(){SETTINGS=await api('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(gatherSettings())});applySecretFieldState();log('Settings saved');await loadCacheInfo()}
async function useH3StorageAsMain(){if(!confirm('Use the H3/newer bucket for future global uploads and main-bucket browsing?\n\nThis changes routing only. It does not copy existing objects, and other endpoints must have the newer network volume attached before they can use its volume paths.'))return;try{SETTINGS=await api('/api/settings/use-h3-storage-as-main',{method:'POST'});applySettings(false);applyH3Settings(SETTINGS.h3||{},SETTINGS.h3_storage||{});set('browser_storage','h3');BROWSER_STORAGE='h3';log('H3 storage is now the main routing profile for future uploads. Existing objects were not moved.');alert('The H3 bucket is now the main routing profile for future uploads. Existing objects were not moved.')}catch(e){log('Storage routing change failed: '+e.message);alert('Storage routing change failed: '+e.message)}}
async function loadCacheInfo(){try{let r=await api('/api/cache/info');let lines=[`App cache: ${r.cache_dir}`,`App cache total: ${r.total_mb} MB`,`SAM outputs: ${r.sam_outputs_mb} MB  ${r.sam_output_dir}`,`Auto-clean: ${r.cleanup_enabled?'on':'off'} / keep ${r.max_age_days} days`,`Hugging Face cache: ${r.hf_cache_dir}`,''];for(const [name,it] of Object.entries(r.items||{}))lines.push(`${name}: ${it.size_mb} MB  ${it.path}`);if($('cache_info'))$('cache_info').textContent=lines.join('\n')}catch(e){if($('cache_info'))$('cache_info').textContent='Cache info error: '+e.message}}
async function cleanCache(force=false){if(force&&!confirm('Clear app cache files now? This removes uploads, temp files, and thumbnails, but not completed outputs, SAM outputs, or Hugging Face model weights.'))return;let r=await api('/api/cache/cleanup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({force})});log(`Cache cleanup removed ${r.removed_files} files / ${r.removed_mb} MB`);await loadCacheInfo()}
async function loadPrompts(){PROMPTS=await api('/api/prompts');for(const id of ['inf_prompt_pick','wan_prompt_pick','h3_prompt_pick']){let s=$(id);if(!s)continue;let current=s.value;s.innerHTML='';let names=Object.keys(PROMPTS).sort();if(!names.length){let o=document.createElement('option');o.value='';o.textContent='No saved prompts yet';s.appendChild(o);continue}for(const n of names){let o=document.createElement('option');o.value=n;o.textContent=n;s.appendChild(o)}if(names.includes(current))s.value=current}}
function loadPrompt(p){let name=$(p+'_prompt_pick').value;if(!name)return;$(p+'_prompt').value=PROMPTS[name]||'';if(p==='h3'){H3_SUBJECT_LAST_BLOCK='';syncH3SubjectDefinitions()}log('Recalled prompt: '+name)}
function randomWanSeed(){let seed=Math.floor(Math.random()*9007199254740991);$('wan_seed').value=String(seed);log('Random Wan seed: '+seed);return seed}
function normalizeWanSeedForSubmit(){let raw=val('wan_seed').trim();if(raw===''||Number(raw)<0){let seed=randomWanSeed();if(raw!==''&&Number(raw)<0)log('Seed -1 is not valid for this endpoint; replaced with '+seed);return seed}return Number(raw)||12345}

function updateInfiniteModeUI(){let isVideo=val('inf_input_type')==='video';let isMulti=val('inf_person_count')==='multi';$('inf_image_path').closest('.drop').classList.toggle('hiddenRow',isVideo);$('inf_video_path').closest('.drop').classList.toggle('hiddenRow',!isVideo);$('inf_audio2_drop').classList.toggle('hiddenRow',!isMulti);$('inf_audio2_tool')?.classList.toggle('hiddenRow',!isMulti);$('inf_audio2_preview').classList.toggle('hidden',!isMulti||!$('inf_audio2_preview').src);$('inf_video_preview').classList.toggle('hidden',!isVideo||!$('inf_video_preview').src);$('inf_image_preview').classList.toggle('hidden',isVideo||!$('inf_image_preview').src)}
async function savePrompt(p){let name=prompt('Save prompt as:');if(!name)return;PROMPTS=await api('/api/prompts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,prompt:$(p+'_prompt').value})});await loadPrompts();$(p+'_prompt_pick').value=name;log('Saved prompt: '+name)}
async function deletePrompt(p){let pick=$(p+'_prompt_pick'),name=pick?.value;if(!name)return alert('Choose a saved prompt first.');if(!confirm(`Delete saved prompt "${name}"?`))return;PROMPTS=await api('/api/prompts/'+encodeURIComponent(name),{method:'DELETE'});await loadPrompts();log('Deleted prompt: '+name)}
function acceptForKind(kind){return kind==='image'?'image/*':kind==='video'?'video/*':kind==='audio'?'audio/*':kind==='lora'?'.safetensors':kind==='media'?'image/*,video/*':'*/*'}
async function uploadFileToTarget(file,target,kind){if(!file)return;if(target==='wan_mask_path')setWanMaskPretrimmed(false);let fd=new FormData();fd.append('file',file);fd.append('kind',kind||'media');let up=await api('/api/upload',{method:'POST',body:fd});$(target).value=up.path;log('Uploaded local copy: '+up.path);previewForInput(target,up.url);if(/^h3_reference\d+_path$/.test(target))syncH3SubjectDefinitions();if(/^h3_reference_(?:video|audio)\d+_path$/.test(target))updateH3ReferenceLabels();await persistSettingsQuietly()}
function openNativePicker(target,kind){let input=document.createElement('input');input.type='file';input.accept=acceptForKind(kind);input.onchange=()=>uploadFileToTarget(input.files?.[0],target,kind);input.click()}
function persistSettingsSoon(){clearTimeout(window.__settingsPersistTimer);window.__settingsPersistTimer=setTimeout(()=>persistSettingsQuietly(),220)}
function setupDrops(){document.querySelectorAll('.drop').forEach(d=>{let input=$(d.dataset.target);if(!d.querySelector('.dropClear')){let clear=document.createElement('button');clear.type='button';clear.className='dropClear secondaryButton';clear.textContent='X';clear.title='Clear this input and remove it from saved payloads';clear.dataset.tip='Clear this input and remove it from saved payloads';clear.onclick=e=>{e.preventDefault();clearPathInput(d.dataset.target)};d.appendChild(clear)}if(!d.querySelector('.dropEdit')){let edit=document.createElement('button');edit.type='button';edit.className='dropEdit secondaryButton';edit.textContent='Edit';edit.title='Open this image in the Image Editor';edit.dataset.tip='Open this image in the Image Editor';edit.onclick=e=>{e.preventDefault();imageEditorLoadFromInput(d.dataset.target)};d.appendChild(edit)}if(!d.querySelector('.dropBrowse')){let btn=document.createElement('button');btn.type='button';btn.className='dropBrowse secondaryButton';btn.textContent='Browse';btn.dataset.tip='Choose a local file';btn.onclick=e=>{e.preventDefault();openNativePicker(d.dataset.target,d.dataset.kind||'media')};d.appendChild(btn)}if(input&&!input.dataset.recentBound){input.dataset.recentBound='1';input.addEventListener('focus',()=>showPathRecent(input));input.addEventListener('click',()=>showPathRecent(input));input.addEventListener('input',()=>{if(input.id==='wan_mask_path')setWanMaskPretrimmed(false);showPathRecent(input);if(!input.value.trim())clearPathInput(input.id,{persist:false,announce:false});markPayloadStateDirty(input.id);persistSettingsSoon()});input.addEventListener('change',async()=>{if(input.id==='wan_mask_path')setWanMaskPretrimmed(false);let p=input.value.trim();if(p){try{previewForInput(input.id,await previewUrlForPath(p))}catch(e){log('Preview failed for '+input.id+': '+e.message)}}else clearPathInput(input.id,{persist:false,announce:false});markPayloadStateDirty(input.id);await persistSettingsQuietly()})}d.ondragover=e=>{e.preventDefault();d.classList.add('drag')};d.ondragleave=()=>d.classList.remove('drag');d.ondrop=async e=>{e.preventDefault();d.classList.remove('drag');await uploadFileToTarget(e.dataTransfer.files[0],d.dataset.target,d.dataset.kind||'media')}});document.addEventListener('click',e=>{if(!e.target.closest?.('.pathRecentMenu')&&!e.target.closest?.('.pathRecentTrigger')&&!e.target.closest?.('.drop input'))hidePathRecent()})}
function tabForPathInput(id){if(String(id).startsWith('inf_'))return'inf';if(String(id).startsWith('wan_'))return'wan';if(String(id).startsWith('h3_'))return'h3';if(String(id).startsWith('samimate_'))return'samimate';if(String(id).startsWith('sam_'))return'sam';return''}
function markPayloadStateDirty(id){let tab=tabForPathInput(id),el=tab&&$(tab+'_validation'),ta=tab&&$(tab+'_payload_preview');if(el)el.innerHTML='<div class="validationItem warn">Inputs changed. Validate again before submitting.</div>';if(ta)ta.value=''}
function clearPathInput(id,{persist=true,announce=true}={}){let el=$(id);if(!el)return;el.value='';hidePathRecent();clearPreviewForInput(id);if(id==='wan_mask_path'){setWanMaskPretrimmed(false);clearWanAlignedMaskPreview()}if(id==='wan_video_path')clearWanAlignedMaskPreview();if(id==='sam_source_path'&&typeof SAM!=='undefined')SAM.result={};markPayloadStateDirty(id);if(persist)persistSettingsQuietly();if(announce)log('Cleared '+id)}
function clearMediaEl(el){if(!el)return;el.classList.add('hidden');el.removeAttribute('src');if(el.load)try{el.load()}catch{}}
function clearWanAlignedMaskPreview(){clearMediaEl($('wan_aligned_mask_preview'));clearMediaEl($('wan_aligned_mask_image_preview'));set('wan_aligned_mask_path','')}
function clearBasePreviewForInput(id){if(id==='inf_image_path')clearMediaEl($('inf_image_preview'));if(id==='inf_video_path'){clearMediaEl($('inf_video_preview'));updateTimeline('inf')}if(id==='inf_audio_path'){clearMediaEl($('inf_audio_preview'));updateAudioTimeline('inf_audio')}if(id==='inf_audio2_path'){clearMediaEl($('inf_audio2_preview'));updateAudioTimeline('inf_audio2')}if(id==='wan_image_path')clearMediaEl($('wan_image_preview'));if(id==='wan_video_path'){clearMediaEl($('wan_video_preview'));clearMediaEl($('wan_trim_video_preview'));clearWanAlignedMaskPreview();drawVideoToPoints();updateTimeline('wan')}if(id==='wan_mask_path')clearWanAlignedMaskPreview();if(id==='wan_masked_video_path')clearMediaEl($('wan_masked_video_preview'));if(id==='samimate_image_path')clearMediaEl($('samimate_image_preview'));if(id==='samimate_video_path'){clearMediaEl($('samimate_video_preview'));clearMediaEl($('samimate_trim_video_preview'));updateTimeline('samimate')}if(id==='sam_source_path'&&typeof SAM!=='undefined'){let c=$('sam_canvas');if(c){let ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height)}SAM.img=null;SAM.frame=null;SAM.points={positive:[],negative:[]};SAM.box=null}updateInfiniteModeUI()}
function previewBaseForInput(id,url){rememberPathForInput(id,val(id));if(!val(id).trim()||!url){clearPreviewForInput(id);return}if(id==='inf_image_path'){set('inf_input_type','image');let img=$('inf_image_preview');img.src=url;img.classList.remove('hidden');$('inf_video_preview').classList.add('hidden')}if(id==='wan_image_path'||id==='samimate_image_path'){let img=id==='samimate_image_path'?$('samimate_image_preview'):$('wan_image_preview');if(img){img.src=url;img.classList.remove('hidden')}}if(id==='inf_video_path'||id==='wan_video_path'||id==='wan_masked_video_path'||id==='samimate_video_path'){if(id==='inf_video_path')set('inf_input_type','video');let v=id==='inf_video_path'?$('inf_video_preview'):(id==='wan_masked_video_path'?$('wan_masked_video_preview'):(id==='samimate_video_path'?$('samimate_video_preview'):$('wan_video_preview')));v.src=url;v.classList.remove('hidden');v.load();if(id==='wan_video_path'){applyWanSourceDefaults(val(id));let tv=$('wan_trim_video_preview');if(tv){tv.src=url;tv.classList.remove('hidden');tv.load();tv.onloadedmetadata=()=>{updateTimeline('wan');applyWanSourceDefaults(val(id));refreshWanPointFrameToTrimStart()}}}if(id==='samimate_video_path'){applySamimateSourceDefaults(val(id));if($('samimate_trim_video_preview')){let tv=$('samimate_trim_video_preview');tv.src=url;tv.classList.remove('hidden');tv.load();tv.onloadedmetadata=()=>{updateTimeline('samimate');applySamimateSourceDefaults(val(id))}}}if(id==='inf_video_path')$('inf_image_preview').classList.add('hidden');v.onloadedmetadata=()=>{if(id==='wan_video_path'){applyWanSourceDefaults(val(id));refreshWanPointFrameToTrimStart()}if(id!=='wan_masked_video_path')updateTimeline(id==='inf_video_path'?'inf':(id==='samimate_video_path'?'samimate':'wan'));if(id==='samimate_video_path')applySamimateSourceDefaults(val(id))}}if(id==='inf_audio_path')loadAudioTrimPreview('inf_audio',url);if(id==='inf_audio2_path')loadAudioTrimPreview('inf_audio2',url);if(id==='sam_source_path'&&!RESTORING_PREVIEWS){samLoadFrame()}updateInfiniteModeUI()}
function clearPreviewForInput(id){let sm=id.match(/^samimate_reference(\d+)_path$/);if(sm)clearMediaEl($(`samimate_reference${sm[1]}_preview`));clearBasePreviewForInput(id);if(id==='h3_first_frame_path')clearMediaEl($('h3_first_frame_preview'));if(id==='h3_last_frame_path')clearMediaEl($('h3_last_frame_preview'));let match=id.match(/^h3_reference(\d+)_path$/);if(match){clearMediaEl($(`h3_reference${match[1]}_preview`));$(`h3_reference${match[1]}_empty`)?.classList.remove('hidden')}let mediaMatch=id.match(/^h3_reference_(video|audio)(\d+)_path$/);if(mediaMatch){clearMediaEl($(`h3_reference_${mediaMatch[1]}${mediaMatch[2]}_preview`));$(`h3_reference_${mediaMatch[1]}${mediaMatch[2]}_empty`)?.classList.remove('hidden');updateH3ReferenceLabels()}}
function previewForInput(id,url){let sm=id.match(/^samimate_reference(\d+)_path$/);if(sm){let img=$(`samimate_reference${sm[1]}_preview`);if(img){img.src=url;img.classList.remove('hidden')}}previewBaseForInput(id,url);if(id==='h3_first_frame_path'||id==='h3_last_frame_path'){let img=$(id==='h3_first_frame_path'?'h3_first_frame_preview':'h3_last_frame_preview');if(img){img.src=url;img.classList.remove('hidden')}}let match=id.match(/^h3_reference(\d+)_path$/);if(match){let img=$(`h3_reference${match[1]}_preview`);if(img){img.src=url;img.classList.remove('hidden');$(`h3_reference${match[1]}_empty`)?.classList.add('hidden')}}let mediaMatch=id.match(/^h3_reference_(video|audio)(\d+)_path$/);if(mediaMatch){let media=$(`h3_reference_${mediaMatch[1]}${mediaMatch[2]}_preview`);if(media){media.src=url;media.classList.remove('hidden');media.load();$(`h3_reference_${mediaMatch[1]}${mediaMatch[2]}_empty`)?.classList.add('hidden');updateH3ReferenceLabels()}}}
function setupImageEditor(){let c=$('image_editor_canvas');if(!c)return;const pointer=(e)=>{let r=c.getBoundingClientRect(),x=(e.clientX-r.left)*(c.width/r.width),y=(e.clientY-r.top)*(c.height/r.height);return imageEditorCanvasToImage(x,y)};c.addEventListener('pointerdown',e=>{if(!IMAGE_EDITOR.img)return;let tool=val('image_editor_tool');let p=pointer(e);if(tool==='crop'){IMAGE_EDITOR.drag={type:'crop',start:p,cur:p}}else if(tool.startsWith('lasso')){IMAGE_EDITOR.lasso=[p];IMAGE_EDITOR.drag={type:'lasso'}}else if(tool.startsWith('sam')){let neg=e.shiftKey||e.button===2;IMAGE_EDITOR.points[neg?'negative':'positive'].push(p);imageEditorDraw()}c.setPointerCapture(e.pointerId);e.preventDefault()});c.addEventListener('pointermove',e=>{if(!IMAGE_EDITOR.drag)return;let p=pointer(e);if(IMAGE_EDITOR.drag.type==='crop')IMAGE_EDITOR.drag.cur=p;if(IMAGE_EDITOR.drag.type==='lasso')IMAGE_EDITOR.lasso.push(p);imageEditorDraw()});c.addEventListener('pointerup',e=>{if(!IMAGE_EDITOR.drag)return;if(IMAGE_EDITOR.drag.type==='crop'){let a=IMAGE_EDITOR.drag.start,b=IMAGE_EDITOR.drag.cur;IMAGE_EDITOR.crop={x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),w:Math.abs(a.x-b.x),h:Math.abs(a.y-b.y)}}IMAGE_EDITOR.drag=null;imageEditorDraw();try{c.releasePointerCapture(e.pointerId)}catch{}});c.addEventListener('contextmenu',e=>e.preventDefault());window.addEventListener('resize',()=>imageEditorDraw())}
function imageEditorCanvasToImage(x,y){return{x:Math.max(0,Math.min(IMAGE_EDITOR.img?.naturalWidth||0,(x-IMAGE_EDITOR.ox)/IMAGE_EDITOR.scale)),y:Math.max(0,Math.min(IMAGE_EDITOR.img?.naturalHeight||0,(y-IMAGE_EDITOR.oy)/IMAGE_EDITOR.scale))}}
function imageEditorImageToCanvas(p){return{x:IMAGE_EDITOR.ox+p.x*IMAGE_EDITOR.scale,y:IMAGE_EDITOR.oy+p.y*IMAGE_EDITOR.scale}}
function imageEditorSetTool(tool){set('image_editor_tool',tool);document.querySelectorAll('.paintTool[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));imageEditorDraw()}
async function imageEditorLoad(path,target='',keepOutput=false){path=String(path||'').trim();if(!path)return alert('Choose an image first.');let kind=mediaKindFromPath(path);if(kind!=='image')return alert('Image Editor only supports still images. Use SAM Studio for videos.');let url=cacheBustUrl(await previewUrlForPath(path));return new Promise((resolve,reject)=>{let img=new Image();img.onload=()=>{IMAGE_EDITOR={...IMAGE_EDITOR,img,path,target,crop:null,lasso:[],points:{positive:[],negative:[]},drag:null,output:keepOutput?IMAGE_EDITOR.output:''};set('image_editor_source',path);if(!keepOutput)set('image_editor_output','');showTab('image_editor');imageEditorDraw();$('image_editor_status').textContent=`Loaded ${img.naturalWidth} x ${img.naturalHeight}`;resolve()};img.onerror=()=>{alert('Could not load image preview.');reject(new Error('Could not load image preview'))};img.src=url})}
async function imageEditorLoadFromInput(id){let p=val(id);if(!p)return alert('No image loaded in '+id);await imageEditorLoad(p,id)}
function imageEditorResetView(){IMAGE_EDITOR.crop=null;IMAGE_EDITOR.lasso=[];IMAGE_EDITOR.points={positive:[],negative:[]};imageEditorDraw()}
function imageEditorClearMarks(){IMAGE_EDITOR.crop=null;IMAGE_EDITOR.lasso=[];IMAGE_EDITOR.points={positive:[],negative:[]};imageEditorDraw()}
function imageEditorDraw(){let c=$('image_editor_canvas');if(!c)return;let ctx=c.getContext('2d'),wrap=c.parentElement,w=Math.max(520,Math.floor(wrap?.clientWidth||960)),h=600;c.width=w;c.height=h;ctx.fillStyle='#020617';ctx.fillRect(0,0,w,h);let img=IMAGE_EDITOR.img;if(!img){ctx.fillStyle='#94a3b8';ctx.fillText('Load an image to begin',24,36);return}let s=Math.min(w/img.naturalWidth,h/img.naturalHeight);IMAGE_EDITOR.scale=s;IMAGE_EDITOR.ox=(w-img.naturalWidth*s)/2;IMAGE_EDITOR.oy=(h-img.naturalHeight*s)/2;ctx.drawImage(img,IMAGE_EDITOR.ox,IMAGE_EDITOR.oy,img.naturalWidth*s,img.naturalHeight*s);ctx.lineWidth=2;ctx.strokeStyle='#22c55e';ctx.fillStyle='rgba(34,197,94,.12)';let crop=IMAGE_EDITOR.drag?.type==='crop'?{x:Math.min(IMAGE_EDITOR.drag.start.x,IMAGE_EDITOR.drag.cur.x),y:Math.min(IMAGE_EDITOR.drag.start.y,IMAGE_EDITOR.drag.cur.y),w:Math.abs(IMAGE_EDITOR.drag.start.x-IMAGE_EDITOR.drag.cur.x),h:Math.abs(IMAGE_EDITOR.drag.start.y-IMAGE_EDITOR.drag.cur.y)}:IMAGE_EDITOR.crop;if(crop&&crop.w>1&&crop.h>1){let a=imageEditorImageToCanvas(crop);ctx.fillRect(a.x,a.y,crop.w*s,crop.h*s);ctx.strokeRect(a.x,a.y,crop.w*s,crop.h*s)}if(IMAGE_EDITOR.lasso.length>1){ctx.strokeStyle=val('image_editor_tool')==='lasso_remove'?'#ef4444':'#38bdf8';ctx.beginPath();IMAGE_EDITOR.lasso.forEach((p,i)=>{let q=imageEditorImageToCanvas(p);if(i)ctx.lineTo(q.x,q.y);else ctx.moveTo(q.x,q.y)});ctx.stroke()}for(const [key,color] of [['positive','#22c55e'],['negative','#ef4444']])for(const p of IMAGE_EDITOR.points[key]){let q=imageEditorImageToCanvas(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q.x,q.y,6,0,Math.PI*2);ctx.fill();ctx.fillStyle='#fff';ctx.fillText(key==='positive'?'+':'-',q.x-3,q.y+4)}}
function imageEditorPayload(){let tool=val('image_editor_tool'),samMode=tool==='sam_remove'?'remove':tool==='sam_keep'?'keep':'none',lassoMode=tool==='lasso_remove'?'remove':tool==='lasso_keep'?'keep':'none';return{path:IMAGE_EDITOR.path,crop:IMAGE_EDITOR.crop||{},lasso:IMAGE_EDITOR.lasso||[],lasso_mode:lassoMode,mask_mode:samMode!=='none'?samMode:lassoMode,mask_fill:val('image_editor_mask_fill'),filter:val('image_editor_filter'),filter_amount:+val('image_editor_filter_amount')||1,upscale:+val('image_editor_upscale')||1,sam:{enabled:samMode!=='none',prompt:val('image_editor_sam_prompt'),prompt_mode:'all',points:IMAGE_EDITOR.points,box:IMAGE_EDITOR.crop||{},sam3_python:val('sam3_python'),hf_token:val('sam_hf_token')}}}
async function imageEditorApply(){if(!IMAGE_EDITOR.path)return alert('Load an image first.');let btn=document.activeElement;try{if(btn)btn.disabled=true;$('image_editor_status').textContent='Applying edit...';let stopProgress=startProgressLogger('image_editor','Image Editor apply',estimateSecFor('image_editor',imageEditorPayload()),'Applying edit...');let r;try{r=await api('/api/image-editor/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(imageEditorPayload())});stopProgress('edit saved')}catch(e){stopProgress('edit failed');throw e}IMAGE_EDITOR.output=r.path;set('image_editor_output',r.path);$('image_editor_status').textContent=`Saved ${r.width} x ${r.height}`;log('Image Editor saved: '+r.path);showOutputPreview('image_editor',[{path:r.path,url:r.url,kind:'image',label:`Edited image ${r.width} x ${r.height}`}],true,{title:'Image Edit'});await imageEditorLoad(r.path,IMAGE_EDITOR.target,true);IMAGE_EDITOR.output=r.path;set('image_editor_output',r.path)}catch(e){$('image_editor_status').textContent='Image edit failed';log('Image Editor error: '+e.message);alert('Image Editor error: '+e.message)}finally{if(btn)btn.disabled=false}}
async function imageEditorSendTo(id){let p=val('image_editor_output')||IMAGE_EDITOR.output||IMAGE_EDITOR.path;if(!p)return alert('Apply an edit first.');set(id,p);previewForInput(id,await previewUrlForPath(p));await persistSettingsQuietly();log('Image Editor output sent to '+id)}
async function applySamimateSourceDefaults(sourcePath){
  sourcePath=String(sourcePath||'').trim();
  if(!sourcePath)return;
  try{
    let info=await api('/api/media/probe?path='+encodeQS(sourcePath));
    let crop=parseCrop('samimate')||{},size=samimateGenerationBackend()==='h3'&&val('samimate_resolution')==='auto'?samimateAutoSize(crop.w>1?crop.w:info.width,crop.h>1?crop.h:info.height):info;
    let fields=[['samimate_width',size.width,832],['samimate_height',size.height,480],['samimate_fps',info.fps,16]];
    let changed=[];
    for(const [id,next,generic] of fields){
      if(id==='samimate_fps'&&samimateGenerationBackend()==='h3')continue;
      let el=$(id);
      if(!el||!next)continue;
      let cur=String(el.value||'').trim(),last=el.dataset.autoValue||'';
      if(val('samimate_resolution')==='auto'||(id==='samimate_fps'&&(!cur||cur===String(generic)||cur===last))){
        el.value=String(next);
        el.dataset.autoValue=String(next);
        changed.push(`${id.replace('samimate_','')}=${next}`);
      }
    }
    if(changed.length){log('SAMimate source defaults: '+changed.join(', '));await persistSettingsQuietly()}
  }catch(e){
    let v=$('samimate_trim_video_preview')||$('samimate_video_preview');
    if(v?.videoWidth&&v?.videoHeight){
      for(const [id,next,generic] of [['samimate_width',v.videoWidth,832],['samimate_height',v.videoHeight,480]]){
        let el=$(id),cur=String(el?.value||'').trim(),last=el?.dataset.autoValue||'';
        if(el&&(!cur||cur===String(generic)||cur===last)){el.value=String(next);el.dataset.autoValue=String(next)}
      }
      await persistSettingsQuietly();
    }else log('SAMimate source metadata unavailable: '+e.message);
  }
}
async function applyWanSourceDefaults(sourcePath){
  sourcePath=String(sourcePath||'').trim();
  if(!sourcePath)return;
  try{
    let info=await api('/api/media/probe?path='+encodeQS(sourcePath));
    let fields=[['wan_width',info.width,832],['wan_height',info.height,480],['wan_fps',info.fps,16]];
    let changed=[];
    for(const [id,next,generic] of fields){
      let el=$(id);
      if(!el||!next)continue;
      let cur=String(el.value||'').trim(),last=el.dataset.autoValue||'';
      if(!cur||cur==='0'||cur===String(generic)||cur===last){
        el.value=String(next);
        el.dataset.autoValue=String(next);
        changed.push(`${id.replace('wan_','')}=${next}`);
      }
    }
    if(changed.length){log('Wan source defaults: '+changed.join(', '));await persistSettingsQuietly()}
  }catch(e){
    let v=$('wan_video_preview');
    if(v?.videoWidth&&v?.videoHeight){
      for(const [id,next,generic] of [['wan_width',v.videoWidth,832],['wan_height',v.videoHeight,480]]){
        let el=$(id),cur=String(el?.value||'').trim(),last=el?.dataset.autoValue||'';
        if(el&&(!cur||cur==='0'||cur===String(generic)||cur===last)){el.value=String(next);el.dataset.autoValue=String(next)}
      }
      await persistSettingsQuietly();
    }else log('Wan source metadata unavailable: '+e.message);
  }
}
function fitCanvasToMedia(c,mediaW,mediaH,maxW=720,maxH=420){if(!c||!mediaW||!mediaH)return;let parent=c.parentElement||c,available=Math.floor(parent.clientWidth||c.clientWidth||900),w=Math.min(Math.max(240,available),maxW),h=Math.round(w*mediaH/mediaW);if(h>maxH){h=maxH;w=Math.round(h*mediaW/mediaH)}w=Math.max(240,w);h=Math.max(140,h);if(c.width!==w||c.height!==h){c.width=w;c.height=h}}
function buildVideoTools(prefix){let box=document.querySelector(`.videoTools[data-prefix="${prefix}"]`);box.innerHTML=`<div class="videoToolGrid"><div><div class="trimVideoMount" id="${prefix}_video_mount"></div><div class="trimPlayerRow"><button id="${prefix}_video_play_btn" class="trimPlayButton" onclick="toggleVid('${prefix}')" title="Play or pause preview" aria-label="Play or pause">&#9654;</button><div class="timeline" id="${prefix}_timeline" title="Drag green/red handles to trim. Click the bar to scrub."><div class="range"></div><div class="handle start" data-handle="start"></div><div class="handle end" data-handle="end"></div><div class="playhead"></div></div></div><div class="trimMeta" id="${prefix}_trim_meta">No video loaded</div></div><div><div class="fields compact"><label title="Trim start time in seconds">Start<input id="${prefix}_video_start" placeholder="0"></label><label title="Trim end time in seconds">End<input id="${prefix}_video_end" placeholder="full"></label><label title="Limit frames when no trim is active">Frame Cap<input id="${prefix}_video_frame_cap" placeholder="0"></label><label title="Raw preprocess JSON. Advanced users can edit this directly.">Crop / Resize JSON<input id="${prefix}_video_crop" placeholder='{"x":0,"y":0,"w":0,"h":0}'></label></div><div class="resizePanel"><div class="fields compact"><label title="Crop x position in source pixels">X<input id="${prefix}_crop_x" type="number" min="0" placeholder="0"></label><label title="Crop y position in source pixels">Y<input id="${prefix}_crop_y" type="number" min="0" placeholder="0"></label><label title="Crop width in source pixels">Crop W<input id="${prefix}_crop_w" type="number" min="0" placeholder="full"></label><label title="Crop height in source pixels">Crop H<input id="${prefix}_crop_h" type="number" min="0" placeholder="full"></label><label title="Optional final resize width after crop">Resize W<input id="${prefix}_resize_w" type="number" min="0" placeholder="keep"></label><label title="Optional final resize height after crop">Resize H<input id="${prefix}_resize_h" type="number" min="0" placeholder="keep"></label></div><div class="videoButtons"><button onclick="fillCropFull('${prefix}')" title="Use the full source frame">Full Frame</button><button onclick="applyCropFields('${prefix}')" title="Write crop/resize fields to JSON">Apply Crop/Resize</button><button onclick="clearCropFields('${prefix}')" title="Clear crop and resize">Clear</button></div></div></div></div><p class="hint">Trim overrides Frame Cap. Crop runs before resize during local preprocessing.</p>`;setupTimeline(prefix);setupCropSync(prefix)}
function mountTrimVideoPreview(prefix){let mount=$(prefix+'_video_mount'),v=prefix==='inf' ? $('inf_video_preview') : null;if((prefix==='wan'||prefix==='samimate')&&mount){let id=prefix==='wan'?'wan_trim_video_preview':'samimate_trim_video_preview';v=$(id);if(!v){v=document.createElement('video');v.id=id;v.controls=true;v.muted=true;v.className='mediaPreview trimVideoPreview hidden';mount.appendChild(v)}}if(mount&&v&&!mount.contains(v)){mount.appendChild(v);v.classList.add('trimVideoPreview')}}
function buildAudioTools(prefix){let box=document.querySelector(`.audioTools[data-prefix="${prefix}"]`);if(!box)return;let label=box.dataset.label||'Audio';box.innerHTML=`<div class="audioTool"><div class="cardTitleRow"><h4>${label}</h4></div><div class="trimPlayerRow"><button id="${prefix}_play_btn" class="trimPlayButton" type="button" onclick="toggleAudioTrim('${prefix}')" title="Play or pause audio" aria-label="Play or pause">&#9654;</button><div class="audioWave" id="${prefix}_wave" title="Drag green/red handles to trim. Click the waveform to scrub."><canvas id="${prefix}_wave_canvas"></canvas><div class="range"></div><div class="handle start" data-handle="start"></div><div class="handle end" data-handle="end"></div><div class="playhead"></div></div></div><div class="trimMeta" id="${prefix}_audio_meta">No audio loaded</div><div class="fields compact"><label>Start<input id="${prefix}_start" placeholder="0"></label><label>End<input id="${prefix}_end" placeholder="full"></label></div></div>`;setupAudioTimeline(prefix);bindAudioTrimElement(prefix)}
function audioEl(p){return $(document.querySelector(`.audioTools[data-prefix="${p}"]`)?.dataset.audio)}
function audioPathId(p){return document.querySelector(`.audioTools[data-prefix="${p}"]`)?.dataset.path}
function audioDuration(p){let d=audioEl(p)?.duration;return Number.isFinite(d)&&d>0?d:0}
function audioBounds(p){let d=audioDuration(p),st=clampTime(timeVal(p+'_start',0),d),en=val(p+'_end').trim()?clampTime(timeVal(p+'_end',d),d):d;if(en<st)en=st;return{d,st,en}}
function selectedAudioDuration(p='inf_audio'){let b=audioBounds(p);return b.d?Math.max(0,b.en-b.st):0}
function bindAudioTrimElement(p){let a=audioEl(p);if(!a||a.dataset.trimBound)return;a.dataset.trimBound='1';let redraw=()=>{updateAudioTimeline(p);drawWaveform(p)};['loadedmetadata','durationchange','canplay'].forEach(ev=>a.addEventListener(ev,redraw));['timeupdate','seeked','play','pause','ended'].forEach(ev=>a.addEventListener(ev,()=>updateAudioTimeline(p)))}
function loadAudioTrimPreview(p,url){let a=audioEl(p);if(!a)return;bindAudioTrimElement(p);a.onloadedmetadata=null;a.src=url;a.classList.remove('hidden');a.load();setTimeout(()=>{updateAudioTimeline(p);if(audioDuration(p)>0)drawWaveform(p)},80);setTimeout(()=>{updateAudioTimeline(p);if(audioDuration(p)>0)drawWaveform(p)},500)}
function updateAudioTimeline(p){let a=audioEl(p),wave=$(p+'_wave');if(!wave)return;let {d,st,en}=audioBounds(p),btn=$(p+'_play_btn');if(!d){$(p+'_audio_meta').textContent='No audio loaded';if(btn)btn.innerHTML='&#9654;';return}let cur=clampTime(a.currentTime,d),pct=x=>(x/d*100)+'%';wave.querySelector('.start').style.left=pct(st);wave.querySelector('.end').style.left=pct(en);let r=wave.querySelector('.range');r.style.left=pct(st);r.style.width=((en-st)/d*100)+'%';wave.querySelector('.playhead').style.left=pct(cur);if(btn)btn.innerHTML=a.paused?'&#9654;':'&#10073;&#10073;';$(p+'_audio_meta').textContent=`Playhead ${cur.toFixed(2)}s | ${st.toFixed(2)}s to ${en.toFixed(2)}s / ${d.toFixed(2)}s (${(en-st).toFixed(2)}s selected)`}
function setupAudioTimeline(p){let wave=$(p+'_wave'),drag=null;const setFromEvent=(e,mode)=>{let d=audioDuration(p);if(!d)return;let r=wave.getBoundingClientRect(),t=clampTime((e.clientX-r.left)/r.width*d,d),b=audioBounds(p),a=audioEl(p);if(mode==='start'){$(p+'_start').value=Math.min(t,b.en).toFixed(2)}else if(mode==='end'){$(p+'_end').value=Math.max(t,b.st).toFixed(2)}else if(a){a.currentTime=t}updateAudioTimeline(p)};wave.addEventListener('pointerdown',e=>{drag=e.target.dataset.handle||'scrub';wave.setPointerCapture(e.pointerId);setFromEvent(e,drag)});wave.addEventListener('pointermove',e=>{if(drag)setFromEvent(e,drag)});wave.addEventListener('pointerup',e=>{drag=null;try{wave.releasePointerCapture(e.pointerId)}catch{}});[p+'_start',p+'_end'].forEach(id=>$(id).addEventListener('input',()=>updateAudioTimeline(p)))}
async function drawWaveform(p){let a=audioEl(p),canvas=$(p+'_wave_canvas'),src=a?.src;if(!canvas||!src)return;let rect=canvas.parentElement.getBoundingClientRect(),w=Math.max(420,Math.floor(rect.width)),h=116;canvas.width=w;canvas.height=h;let ctx=canvas.getContext('2d');ctx.fillStyle='#020617';ctx.fillRect(0,0,w,h);try{let buf=await (await fetch(src)).arrayBuffer(),ac=new (window.AudioContext||window.webkitAudioContext)(),decoded=await ac.decodeAudioData(buf.slice(0)),data=decoded.getChannelData(0),step=Math.max(1,Math.floor(data.length/w));ctx.strokeStyle='#38bdf8';ctx.lineWidth=1;ctx.beginPath();for(let x=0;x<w;x++){let min=1,max=-1;for(let j=0;j<step;j++){let v=data[x*step+j]||0;if(v<min)min=v;if(v>max)max=v}ctx.moveTo(x,(1+min)*h/2);ctx.lineTo(x,(1+max)*h/2)}ctx.stroke();ac.close?.()}catch{ctx.strokeStyle='#38bdf8';ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke()}updateAudioTimeline(p)}
function toggleAudioTrim(p){let a=audioEl(p);if(!a)return;if(a.paused)a.play();else a.pause();updateAudioTimeline(p)}function playAudioTrim(p){audioEl(p)?.play()}function pauseAudioTrim(p){audioEl(p)?.pause()}function stopAudioTrim(p){let a=audioEl(p);if(!a)return;let b=audioBounds(p);a.pause();a.currentTime=b.st||0;updateAudioTimeline(p)}function setAudioTrimFromPlayer(p,k){let a=audioEl(p);if(!a)return;$(p+'_'+k).value=a.currentTime.toFixed(2);updateAudioTimeline(p)}function resetAudioTrim(p){set(p+'_start','');set(p+'_end','');updateAudioTimeline(p)}
function vid(prefix){return prefix==='inf' ? $('inf_video_preview') : (prefix==='samimate' ? $('samimate_trim_video_preview') : ($('wan_trim_video_preview')||$('wan_video_preview')))}
function videoDuration(p){let d=vid(p).duration;return Number.isFinite(d)&&d>0?d:0}
function timeVal(id,fallback=0){let n=parseFloat(val(id));return Number.isFinite(n)?n:fallback}
function clampTime(t,d){return Math.max(0,Math.min(Number(t)||0,d||0))}
function toggleVid(p){let v=vid(p);if(!v)return;if(v.paused)v.play();else v.pause();updateTimeline(p)}function playVid(p){vid(p).play()}function pauseVid(p){vid(p).pause()}function stopVid(p){let v=vid(p),b=trimBounds(p);v.pause();v.currentTime=b.st||0;updateTimeline(p)}function setTrimFromVideo(p,k){$(p+'_video_'+k).value=vid(p).currentTime.toFixed(2);updateTimeline(p)}function resetTrim(p){$(p+'_video_start').value='';$(p+'_video_end').value='';updateTimeline(p)}
function trimBounds(p){let d=videoDuration(p),st=clampTime(timeVal(p+'_video_start',0),d),en=val(p+'_video_end').trim()?clampTime(timeVal(p+'_video_end',d),d):d;if(en<st)en=st;return{d,st,en}}
function updateTimeline(p){let v=vid(p),tl=$(p+'_timeline');if(!tl)return;let {d,st,en}=trimBounds(p),btn=$(p+'_video_play_btn');if(!d){$(''+p+'_trim_meta').textContent='No video loaded';if(btn)btn.innerHTML='&#9654;';return}let cur=clampTime(v.currentTime,d),pct=x=>(x/d*100)+'%';tl.querySelector('.start').style.left=pct(st);tl.querySelector('.end').style.left=pct(en);let r=tl.querySelector('.range');r.style.left=pct(st);r.style.width=((en-st)/d*100)+'%';tl.querySelector('.playhead').style.left=pct(cur);if(btn)btn.innerHTML=v.paused?'&#9654;':'&#10073;&#10073;';$(p+'_trim_meta').textContent=`Playhead ${cur.toFixed(2)}s | ${st.toFixed(2)}s to ${en.toFixed(2)}s / ${d.toFixed(2)}s (${(en-st).toFixed(2)}s selected)`}
function markWanMaskAlignmentDirty(){if($('wan_aligned_mask_path')?.value)set('wan_aligned_mask_path','Aligned mask preview is stale. Click Preview aligned mask again.')}
function setupTimeline(p){let tl=$(p+'_timeline'),drag=null;const setFromEvent=(e,mode)=>{let d=videoDuration(p);if(!d)return;let r=tl.getBoundingClientRect(),t=clampTime((e.clientX-r.left)/r.width*d,d),b=trimBounds(p);if(mode==='start'){$(p+'_video_start').value=Math.min(t,b.en).toFixed(2)}else if(mode==='end'){$(p+'_video_end').value=Math.max(t,b.st).toFixed(2)}else{vid(p).currentTime=t}updateTimeline(p);if(p==='wan'){markWanMaskAlignmentDirty();refreshWanPointFrameToTrimStart()}};tl.addEventListener('pointerdown',e=>{drag=e.target.dataset.handle||'scrub';tl.setPointerCapture(e.pointerId);setFromEvent(e,drag)});tl.addEventListener('pointermove',e=>{if(drag)setFromEvent(e,drag)});tl.addEventListener('pointerup',e=>{drag=null;try{tl.releasePointerCapture(e.pointerId)}catch{}});[p+'_video_start',p+'_video_end'].forEach(id=>$(id).addEventListener('input',()=>{updateTimeline(p);if(p==='wan'){markWanMaskAlignmentDirty();refreshWanPointFrameToTrimStart()}}))}
function setupCropSync(p){$(p+'_video_crop').addEventListener('change',()=>{loadCropFields(p);if(p==='wan')markWanMaskAlignmentDirty()});['crop_x','crop_y','crop_w','crop_h','resize_w','resize_h'].forEach(s=>$(p+'_'+s).addEventListener('input',()=>{applyCropFields(p,false);if(p==='wan')markWanMaskAlignmentDirty()}))}
function fillCropFull(p){let v=vid(p);if(!v.videoWidth)return alert('Load a video first.');set(p+'_crop_x',0);set(p+'_crop_y',0);set(p+'_crop_w',v.videoWidth);set(p+'_crop_h',v.videoHeight);applyCropFields(p)}
function cropNum(id){let n=parseInt(val(id),10);return Number.isFinite(n)&&n>0?n:0}
function applyCropFields(p,announce=true){let crop={x:cropNum(p+'_crop_x'),y:cropNum(p+'_crop_y'),w:cropNum(p+'_crop_w'),h:cropNum(p+'_crop_h')},rw=cropNum(p+'_resize_w'),rh=cropNum(p+'_resize_h');if(rw)crop.resize_w=rw;if(rh)crop.resize_h=rh;if(!crop.w||!crop.h){if(rw||rh){crop={resize_w:rw||0,resize_h:rh||0}}else{set(p+'_video_crop','');return}}set(p+'_video_crop',JSON.stringify(crop));if(announce)log(`Updated ${p} crop/resize: ${JSON.stringify(crop)}`)}
function loadCropFields(p){let c=parseCrop(p)||{};set(p+'_crop_x',c.x||'');set(p+'_crop_y',c.y||'');set(p+'_crop_w',c.w||'');set(p+'_crop_h',c.h||'');set(p+'_resize_w',c.resize_w||'');set(p+'_resize_h',c.resize_h||'')}
function clearCropFields(p){['crop_x','crop_y','crop_w','crop_h','resize_w','resize_h'].forEach(s=>set(p+'_'+s,''));set(p+'_video_crop','');log(`Cleared ${p} crop/resize`)}
setInterval(()=>{updateTimeline('inf');updateTimeline('wan');updateTimeline('samimate');updateAudioTimeline('inf_audio');updateAudioTimeline('inf_audio2')},250);
function parseCrop(p){try{return JSON.parse($(p+'_video_crop').value||'null')}catch{return null}}
function setupPoints(){
  pointFrame.canvas=$('points_canvas');
  pointFrame.ctx=pointFrame.canvas.getContext('2d');
  $('wan_video_preview').addEventListener('loadeddata',refreshWanPointFrameToTrimStart);
  $('wan_video_preview').addEventListener('seeked',drawVideoToPoints);
  pointFrame.canvas.addEventListener('click',e=>{if(e.shiftKey)addPoint(e,true);else addPoint(e,false)});
  pointFrame.canvas.addEventListener('contextmenu',e=>{e.preventDefault();addPoint(e,true)});
}

function refreshWanPointFrameToTrimStart(){
  let v=$('wan_video_preview');
  if(!v||!v.videoWidth)return;
  let b=trimBounds('wan'),target=b.st||0;
  if(Number.isFinite(target)&&Math.abs((v.currentTime||0)-target)>0.04){
    v.currentTime=target;
  }else{
    drawVideoToPoints();
  }
}

function drawVideoToPoints(){
  let v=$('wan_video_preview'),c=pointFrame.canvas,ctx=pointFrame.ctx;
  if(!v.videoWidth)return;
  fitCanvasToMedia(c,v.videoWidth,v.videoHeight,720,380);
  ctx.fillStyle='#020617';ctx.fillRect(0,0,c.width,c.height);
  let s=Math.min(c.width/v.videoWidth,c.height/v.videoHeight),
      w=v.videoWidth*s,h=v.videoHeight*s,x=(c.width-w)/2,y=(c.height-h)/2;
  ctx.drawImage(v,x,y,w,h);
  pointFrame.box={x,y,w,h,s,iw:v.videoWidth,ih:v.videoHeight};
  drawPoints();
}

function cleanPoint(p){
  return {x:Math.round(Number(p.x)*1000)/1000,y:Math.round(Number(p.y)*1000)/1000};
}

function currentPointPayload(){
  let pos=points.positive.map(cleanPoint);
  let neg=points.negative.map(cleanPoint);
  // WanAnimate's published control-point examples include a dummy negative
  // point of {"x":0,"y":0}. Keep that shape even when the user only adds
  // positive points.
  if(pos.length && !neg.length) neg=[{x:0,y:0}];
  return {positive:pos,negative:neg};
}

function addPoint(e,neg){
  if(!pointFrame.box)return;
  let r=pointFrame.canvas.getBoundingClientRect(),
      x=(e.clientX-r.left)*(pointFrame.canvas.width/r.width),
      y=(e.clientY-r.top)*(pointFrame.canvas.height/r.height),
      b=pointFrame.box;
  if(x<b.x||y<b.y||x>b.x+b.w||y>b.y+b.h)return;
  let pt={x:(x-b.x)/b.s,y:(y-b.y)/b.s};
  let key=neg?'negative':'positive';
  points[key].push(pt);
  pointFrame.history.push(key);
  drawVideoToPoints();
}

function undoPoint(){
  let key=pointFrame.history.pop();
  if(!key){
    if(points.negative.length)key='negative';
    else if(points.positive.length)key='positive';
  }
  if(key&&points[key]?.length)points[key].pop();
  drawVideoToPoints();
  log('Undid last control point');
}

function drawPoints(){
  let ctx=pointFrame.ctx,b=pointFrame.box;
  if(!b)return;
  let i=1;
  for(const p of points.positive){drawPt(ctx,b,p,'#22c55e','+'+(i++))}
  i=1;
  for(const p of points.negative){drawPt(ctx,b,p,'#ef4444','-'+(i++))}
  $('points_status').textContent=` editor +${points.positive.length} / -${points.negative.length}`;
}

function drawPt(ctx,b,p,col,t){
  let x=b.x+p.x*b.s,y=b.y+p.y*b.s;
  ctx.fillStyle=col;ctx.strokeStyle='white';ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.fill();ctx.stroke();
  ctx.fillStyle='white';ctx.fillText(t,x+10,y-10);
}

function applyPointsToPayload(){
  let payload=currentPointPayload();
  $('wan_points_store').value=JSON.stringify(payload);
  $('wan_coordinates').value=JSON.stringify(payload.positive);
  $('wan_neg_coordinates').value=JSON.stringify(payload.negative);
  $('wan_control_points').checked=true;
  $('points_status').textContent=` applied +${payload.positive.length} / -${payload.negative.length}`;
  log(`Applied control points to payload: +${payload.positive.length} / -${payload.negative.length}`);
}

function parsePointJson(id,fallback){
  let raw=val(id).trim();
  if(!raw)return fallback;
  return JSON.parse(raw);
}

function readPointPayloadFromFields(){
  let pointsStoreRaw=val('wan_points_store').trim();
  let coordinatesRaw=val('wan_coordinates').trim();
  let negRaw=val('wan_neg_coordinates').trim();

  if(!coordinatesRaw && pointsStoreRaw){
    let ps=JSON.parse(pointsStoreRaw);
    coordinatesRaw=JSON.stringify(ps.positive||[]);
    negRaw=JSON.stringify(ps.negative||[]);
  }
  if(!pointsStoreRaw && coordinatesRaw){
    let pos=JSON.parse(coordinatesRaw||'[]');
    let neg=negRaw?JSON.parse(negRaw):[];
    pointsStoreRaw=JSON.stringify({positive:pos,negative:neg});
  }
  return {
    points_store_raw:pointsStoreRaw,
    coordinates_raw:coordinatesRaw,
    neg_coordinates_raw:negRaw
  };
}

function loadPayloadToPoints(){
  try{
    let raw=readPointPayloadFromFields();
    let pos=raw.coordinates_raw?JSON.parse(raw.coordinates_raw):[];
    let neg=raw.neg_coordinates_raw?JSON.parse(raw.neg_coordinates_raw):[];
    points={positive:pos,negative:neg};
    pointFrame.history=[];
    drawVideoToPoints();
    log(`Loaded payload into editor: +${pos.length} / -${neg.length}`);
  }catch(e){log('Point payload JSON error: '+e.message)}
}

function clearPointPayload(){
  $('wan_points_store').value='';
  $('wan_coordinates').value='';
  $('wan_neg_coordinates').value='';
  log('Cleared point payload fields');
}

function clearPoints(){
  points={positive:[],negative:[]};
  pointFrame.history=[];
  drawVideoToPoints();
}
async function runInfinite(){let btn=$('inf_run_button'),status=$('inf_status');try{await saveSettings();let inputType=val('inf_input_type'),personCount=val('inf_person_count');if(inputType==='image'&&!val('inf_image_path').trim())return alert('InfiniteTalk image mode requires an image.');if(inputType==='video'&&!val('inf_video_path').trim())return alert('InfiniteTalk video mode requires a video.');if(!val('inf_audio_path').trim())return alert('InfiniteTalk requires an audio file.');if(personCount==='multi'&&!val('inf_audio2_path').trim())return alert('Multi mode requires Audio 2.');if(btn){btn.disabled=true;btn.textContent='Submitting...'}if(status)status.textContent='Preparing payload...';let p={settings:payloadSettings(),delivery_mode:val('delivery_mode'),input_type:inputType,person_count:personCount,image_path:val('inf_image_path'),video_path:val('inf_video_path'),audio_path:val('inf_audio_path'),audio2_path:val('inf_audio2_path'),audio_start:val('inf_audio_start'),audio_end:val('inf_audio_end'),audio2_start:val('inf_audio2_start'),audio2_end:val('inf_audio2_end'),video_start:val('inf_video_start'),video_end:val('inf_video_end'),video_frame_cap:val('inf_video_frame_cap'),video_crop:parseCrop('inf'),prompt:val('inf_prompt'),advanced_json:val('inf_advanced'),width:+val('inf_width'),height:+val('inf_height'),max_frame:val('inf_max_frame'),sync_to_audio:chk('inf_sync_to_audio'),force_offload:chk('inf_force_offload'),network_volume:chk('inf_network_volume')};if(p.sync_to_audio){let dur=selectedAudioDuration('inf_audio');if(dur>0){if(inputType==='video'){let st=timeVal('inf_video_start',0);p.video_start=String(st);p.video_end=(st+dur).toFixed(2);p.video_frame_cap=''}else if(!p.max_frame){p.max_frame=Math.max(1,Math.ceil(dur*24));}log(`InfiniteTalk length matched to audio: ${dur.toFixed(2)}s`)}}await submitRun('/api/run/infinitetalk',p,'inf');if(status)status.textContent='Job submitted'}catch(e){log('InfiniteTalk ERROR: '+e.message);alert('InfiniteTalk error: '+e.message);if(status)status.textContent='Error'}finally{if(btn){btn.disabled=false;btn.textContent='Run InfiniteTalk'}}}
async function runWan(){
  let btn=$('wan_run_button'),status=$('wan_status');
  try{
    await saveSettings();
    if(!val('wan_image_path').trim())return alert('WanAnimate needs a reference image.');
    if(!val('wan_video_path').trim())return alert('WanAnimate needs a reference video.');
    if(chk('wan_mask_edit')&&!val('wan_mask_path').trim())return alert('Mask edit mode needs a subject mask.');
    if(chk('wan_control_points') && points.positive.length)applyPointsToPayload();
    if(btn){btn.disabled=true;btn.textContent='Submitting...'}
    if(status)status.textContent='Preparing aligned media...';
    let pointPayload=readPointPayloadFromFields();
    let p={
      settings:payloadSettings(),delivery_mode:val('delivery_mode'),wan_delivery_override:val('wan_delivery_override'),
      image_path:val('wan_image_path'),video_path:val('wan_video_path'),mask_path:val('wan_mask_path'),masked_video_path:val('wan_masked_video_path'),mask_pretrimmed:wanMaskPretrimmed(),
      video_start:val('wan_video_start'),video_end:val('wan_video_end'),video_frame_cap:val('wan_video_frame_cap'),video_crop:parseCrop('wan'),
      prompt:val('wan_prompt'),negative_prompt:val('wan_negative_prompt'),advanced_json:val('wan_advanced'),
      mode:val('wan_mode'),width:intFieldOrDefault('wan_width',0),height:intFieldOrDefault('wan_height',0),seed:normalizeWanSeedForSubmit(),fps:+val('wan_fps'),cfg:+val('wan_cfg'),steps:+val('wan_steps'),
      pose_estimation:chk('wan_pose'),face_detection:chk('wan_face'),mask_editing:chk('wan_mask_edit'),network_volume:chk('wan_network_volume'),
      control_points_enabled:chk('wan_control_points'),
      ...pointPayload
    };
    await submitRun('/api/run/wananimate',p,'wan');
    if(status)status.textContent='Job submitted';
  }catch(e){log('WanAnimate ERROR: '+e.message);alert('WanAnimate error: '+e.message);if(status)status.textContent='Error'}
  finally{if(btn){btn.disabled=false;btn.textContent='Run WanAnimate'}}
}
async function runH3(){
  let btn=$('h3_run_button'),status=$('h3_status');
  if(btn){btn.disabled=true;btn.textContent='Submitting...'}
  try{
    syncH3SubjectDefinitions();
    await saveSettings();
    let p={settings:payloadSettings(),h3_endpoint_id:val('h3_endpoint_id'),...h3Settings()};
    if(!p.prompt.trim())return alert('MiniMax H3 needs a prompt.');
    if(p.task==='fl2v'&&!p.first_frame_path.trim())return alert('H3 FL2V needs a first frame.');
    if(p.task==='r2v'&&!(p.reference_paths.length+p.reference_video_paths.length+p.reference_audio_paths.length))return alert('H3 R2VA needs at least one image, video, or audio reference.');
    try{p.loras=h3Loras(true)}catch(e){return alert(e.message)}
    if(status)status.textContent='Preparing MiniMax H3 assets...';
    await submitRun('/api/run/h3',p,'h3');
    if(status)status.textContent='Job submitted';
  }catch(e){log('MiniMax H3 ERROR: '+e.message);alert('MiniMax H3 error: '+e.message);if(status)status.textContent='Error'}
  finally{if(btn){btn.disabled=false;btn.textContent='Run MiniMax H3'}}
}
async function previewAlignedWanMask(){
  try{
    if(!val('wan_video_path').trim())return alert('Load a Wan reference video first.');
    if(!val('wan_mask_path').trim())return alert('Load a Wan subject mask first.');
    set('wan_aligned_mask_path','Preparing aligned mask preview...');
    let r=await api('/api/wan/align-mask-preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentPayload('wan'))});
    set('wan_aligned_mask_path',r.path||'');
    clearMediaEl($('wan_aligned_mask_preview'));clearMediaEl($('wan_aligned_mask_image_preview'));
    let url=cacheBustUrl(r.url||('/api/file?path='+encodeQS(r.path)));
    if(r.kind==='image'){let img=$('wan_aligned_mask_image_preview');img.src=url;img.classList.remove('hidden')}
    else{let v=$('wan_aligned_mask_preview');v.src=url;v.classList.remove('hidden');v.load()}
    let mp=r.mask_probe||{},vp=r.video_probe||{};
    log(`Aligned Wan mask preview: ${r.path}`);
    log(`Aligned mask ${mp.width||'?'}x${mp.height||'?'} ${mp.duration_seconds?mp.duration_seconds+'s':''}; video ${vp.width||'?'}x${vp.height||'?'} ${vp.duration_seconds?vp.duration_seconds+'s':''}`);
  }catch(e){
    set('wan_aligned_mask_path','Aligned mask preview failed');
    log('Aligned Wan mask error: '+e.message);
    alert('Aligned Wan mask error: '+e.message);
  }
}
async function copyWanToSamimate(){
  for(const [src,dst] of [['wan_video_path','samimate_video_path'],['wan_image_path','samimate_image_path'],['wan_prompt','samimate_prompt'],['wan_negative_prompt','samimate_negative_prompt'],['wan_advanced','samimate_advanced'],['wan_mode','samimate_mode'],['wan_delivery_override','samimate_delivery_override'],['wan_width','samimate_width'],['wan_height','samimate_height'],['wan_seed','samimate_seed'],['wan_fps','samimate_fps'],['wan_cfg','samimate_cfg'],['wan_steps','samimate_steps'],['wan_video_start','samimate_video_start'],['wan_video_end','samimate_video_end'],['wan_video_frame_cap','samimate_video_frame_cap'],['wan_video_crop','samimate_video_crop']])set(dst,val(src));
  for(const [src,dst] of [['wan_pose','samimate_pose'],['wan_face','samimate_face'],['wan_mask_edit','samimate_mask_edit'],['wan_network_volume','samimate_network_volume'],['wan_control_points','samimate_use_wan_points']])setChk(dst,chk(src));
  loadCropFields('samimate');
  for(const id of ['samimate_video_path','samimate_image_path']){let p=val(id).trim();if(p)try{previewForInput(id,await previewUrlForPath(p))}catch(e){log('SAMimate preview failed: '+e.message)}}
  await persistSettingsQuietly();
  log('Copied current WanAnimate inputs and options into SAMimate');
}
function fmtDurationSec(sec){sec=Math.max(0,Math.round(Number(sec)||0));let m=Math.floor(sec/60),s=sec%60;return m?`${m}m ${String(s).padStart(2,'0')}s`:`${s}s`}
function targetLabel(target){return target==='inf'?'InfiniteTalk':target==='wan'?'WanAnimate':target==='h3'?'MiniMax H3':target==='samimate'?'SAMimate':target==='sam'?'SAM Studio':target==='tts'?'Text to Speech':target==='image_editor'?'Image Editor':'Job'}
function targetStatusEl(target){return $(target+'_status')||$(target+'_tts_status')||$(target+'_editor_status')}
function progressHistoryKey(target){return 'progress_history_'+target}
function progressHistory(target){try{return JSON.parse(localStorage.getItem(progressHistoryKey(target))||'[]').filter(x=>Number.isFinite(Number(x))&&Number(x)>0)}catch{return[]}}
function rememberProgressDuration(target,seconds){seconds=Number(seconds)||0;if(seconds<=5)return;let items=progressHistory(target);items.unshift(seconds);localStorage.setItem(progressHistoryKey(target),JSON.stringify(items.slice(0,8)))}
function averageProgressDuration(target){let items=progressHistory(target);return items.length?items.reduce((a,b)=>a+Number(b),0)/items.length:0}
function payloadFrameGuess(target,p={}){
  if(target==='wan'||target==='samimate'){
    let fps=Number(p.fps||val(target==='wan'?'wan_fps':'samimate_fps')||16),cap=Number(p.video_frame_cap||0),start=Number(p.video_start||0),end=Number(p.video_end||0);
    if(cap>0)return cap;if(end>start&&fps>0)return Math.ceil((end-start)*fps);
  }
  if(target==='inf'){
    let max=Number(p.max_frame||0);if(max>0)return max;
    let aud=selectedAudioDuration('inf_audio');if(aud>0)return Math.ceil(aud*24);
  }
  return 0;
}
function fallbackEstimateSec(target,p={}){
  let frames=payloadFrameGuess(target,p),steps=Number(p.steps||0);
  if(target==='inf')return Math.max(420,Math.min(3600,frames?frames*4:900));
  if(target==='wan')return Math.max(300,Math.min(3600,(frames?frames*3:480)+(steps?steps*35:0)));
  if(target==='h3')return Math.max(30,Math.min(1800,120+(Number(p.duration||2)*25)+(steps?steps*8:0)));
  if(target==='samimate')return Math.max(480,Math.min(4800,(frames?frames*5:900)+(steps?steps*35:0)));
  if(target==='sam')return Math.max(120,Math.min(2400,frames?frames*1.4:300));
  if(target==='tts')return 25;
  if(target==='image_editor')return 45;
  return 600;
}
function estimateSecFor(target,p={}){let h=averageProgressDuration(target),f=fallbackEstimateSec(target,p);return h?Math.max(30,(h*0.7)+(f*0.3)):f}
function progressPercent(status,elapsedSec,estimateSec,runSec=0){
  let s=String(status||'').toUpperCase(),base=s==='IN_QUEUE'?Math.min(9,2+elapsedSec/20):s==='IN_PROGRESS'?10:1;
  if(s==='COMPLETED')return 100;if(s==='FAILED')return 0;
  if(s==='IN_QUEUE')return Math.round(base);
  let active=runSec||elapsedSec;
  return Math.max(base,Math.min(95,Math.round(10+(active/Math.max(estimateSec,1))*85)));
}
function progressSummary(target,status,startedAt,estimateSec,r={}){
  let elapsed=(Date.now()-startedAt)/1000,execMs=Number(r.status?.executionTime||0),delayMs=Number(r.status?.delayTime||0),runSec=execMs?execMs/1000:Math.max(0,elapsed-(delayMs?delayMs/1000:0));
  let pct=progressPercent(status,elapsed,estimateSec,runSec),remaining=Math.max(0,estimateSec-(runSec||elapsed));
  let terminal=['COMPLETED','FAILED','CANCELLED','TIMED_OUT'].includes(String(status).toUpperCase());
  let eta=terminal?'done':pct>=95?'finishing':`~${fmtDurationSec(remaining)}`;
  let bits=[`${pct}%`,String(status||'working'),`elapsed ${fmtDurationSec(elapsed)}`,`ETA ${eta}`];
  if(delayMs)bits.push(`queued ${fmtDurationSec(delayMs/1000)}`);
  if(execMs)bits.push(`running ${fmtDurationSec(execMs/1000)}`);
  return bits.join(' | ');
}
function startProgressLogger(target,label,estimateSec,statusText){
  let key=target+'_'+label,startedAt=Date.now(),statusEl=targetStatusEl(target);
  clearInterval(PROGRESS_TIMERS[key]);
  if(statusEl)statusEl.textContent=statusText||`${label}...`;
  log(`${label}: started | ETA ~${fmtDurationSec(estimateSec)}`);
  PROGRESS_TIMERS[key]=setInterval(()=>{
    let elapsed=(Date.now()-startedAt)/1000,pct=progressPercent('IN_PROGRESS',elapsed,estimateSec,elapsed),eta=pct>=95?'finishing':`~${fmtDurationSec(Math.max(0,estimateSec-elapsed))}`;
    if(statusEl)statusEl.textContent=`${label}: ${pct}% | ETA ${eta}`;
    log(`${label}: ${pct}% | elapsed ${fmtDurationSec(elapsed)} | ETA ${eta}`);
  },30000);
  return function stopProgressLogger(finalMsg='complete'){
    clearInterval(PROGRESS_TIMERS[key]);delete PROGRESS_TIMERS[key];
    let elapsed=(Date.now()-startedAt)/1000;rememberProgressDuration(target,elapsed);
    if(statusEl)statusEl.textContent=finalMsg;
    log(`${label}: ${finalMsg} | elapsed ${fmtDurationSec(elapsed)}`);
  }
}
function samimateTrimSummary(){
  let st=val('samimate_video_start')||'0',en=val('samimate_video_end')||'full',cap=samimateEffectiveFrameCap();
  let d=videoDuration('samimate'),dur='';
  let start=timeVal('samimate_video_start',0),end=val('samimate_video_end')?timeVal('samimate_video_end',d||0):(d||0);
  let selected=d&&end>start?end-start:0,fps=Number(val('samimate_fps')||0),frames=selected&&fps?Math.round(selected*fps):0;
  if(selected)dur=` (${selected.toFixed(2)}s selected${frames?`, about ${frames} frames at ${fps}fps`:''})`;
  return `trim ${st} -> ${en}${dur}; frame cap ${cap||'full'}; target ${val('samimate_width')}x${val('samimate_height')} @ ${val('samimate_fps')}fps`;
}
function samimateSetStatus(msg, detail=''){
  let el=$('samimate_status');if(el)el.textContent=msg||'';
  if(msg){
    let elapsed=SAMIMATE_STATE?.startedAt?` | elapsed ${fmtDurationSec((Date.now()-SAMIMATE_STATE.startedAt)/1000)}`:'';
    log('SAMimate: '+msg+elapsed+(detail?' - '+detail:''));
  }
}
function samimateResetProgress(){}
function samimateWanFrameCap(){
  let explicit=intFieldOrDefault('samimate_video_frame_cap',0);
  if(explicit>0)return explicit;
  return samimateEffectiveFrameCap();
}
function samimateEffectiveFrameCap(){
  let cap=intFieldOrDefault('samimate_frame_cap',0);
  if(cap<0)cap=0;
  set('samimate_frame_cap',cap);
  return cap;
}
function samimateWanPayload(maskVideoPath){
  let pointPayload=chk('samimate_use_wan_points')?readPointPayloadFromFields():{};
  let maskFrameCap=samimateEffectiveFrameCap();
  let wanFrameCap=samimateWanFrameCap();
  return {
    settings:payloadSettings(),generation_backend:'wan',delivery_mode:val('delivery_mode'),wan_delivery_override:val('samimate_delivery_override'),source_workflow:'samimate',samimate_mask_frame_cap:maskFrameCap,mask_frame_cap:maskFrameCap,
    image_path:val('samimate_image_path'),video_path:val('samimate_video_path'),mask_path:maskVideoPath,masked_video_path:'',mask_pretrimmed:true,
    video_start:val('samimate_video_start'),video_end:val('samimate_video_end'),video_frame_cap:String(wanFrameCap),video_crop:parseCrop('samimate'),
    prompt:val('samimate_prompt'),negative_prompt:val('samimate_negative_prompt'),advanced_json:val('samimate_advanced'),
    mode:val('samimate_mode'),width:intFieldOrDefault('samimate_width',0),height:intFieldOrDefault('samimate_height',0),seed:+val('samimate_seed')||normalizeWanSeedForSubmit(),fps:+val('samimate_fps'),cfg:+val('samimate_cfg'),steps:+val('samimate_steps'),
    pose_estimation:chk('samimate_pose'),face_detection:chk('samimate_face'),mask_editing:true,network_volume:chk('samimate_network_volume'),
    control_points_enabled:chk('samimate_use_wan_points'),...pointPayload
  }
}
function samimateGenerationBackend(){return val('samimate_generation_backend')==='h3'?'h3':'wan'}
function setupSamimateReferences(){
  let host=$('samimate_extra_references');if(!host)return;
  host.innerHTML=Array.from({length:8},(_,i)=>{let n=i+2;return `<div class="drop" data-target="samimate_reference${n}_path" data-kind="image">Identity view ${n}<input id="samimate_reference${n}_path" placeholder="Optional image path / URL"><img id="samimate_reference${n}_preview" class="mediaPreview hidden" alt="Identity view ${n}"></div>`}).join('');
}
async function samimatePreviewPrompt(){try{let r=await api('/api/samimate/h3-prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(samimateH3Payload())});$('samimate_auto_prompt').textContent=r.prompt}catch(e){alert(e.message)}}
function samimateReferencePaths(){return [val('samimate_image_path'),...Array.from({length:8},(_,i)=>val(`samimate_reference${i+2}_path`))].map(x=>x.trim()).filter(Boolean)}
function updateSamimateBackendUI(){
  let h3=samimateGenerationBackend()==='h3';
  document.querySelectorAll('.samimateWanOnly').forEach(el=>el.classList.toggle('hidden',h3));
  document.querySelectorAll('.samimateH3Only').forEach(el=>el.classList.toggle('hidden',!h3));
  $('samimate_options_title').textContent=h3?'MiniMax H3 Options':'WanAnimate Options';
  $('samimate_backend_hint').textContent=h3?'Ref2VA replaces the SAM area using all identity views. The original background and audio are composited back. Select a 1–15 second trim; H3 uses 24 fps. Prepare masks once, then choose the H3 settings below and send.':'WanAnimate receives the SAM subject mask as its edit area, then the original background is composited back.';
  if(h3)set('samimate_fps',24);
  $('samimate_fps').disabled=h3;
  if($('samimate_run_button'))$('samimate_run_button').textContent=h3?'Send to H3':'Run SAMimate';
  let trimHint=document.querySelector('.videoTools[data-prefix="samimate"] > .hint');if(trimHint)trimHint.textContent=h3?'H3 uses 24 fps for masking. The smaller nonzero frame cap limits the selected trim. The final master retains the source frame rate.':'Trim overrides Frame Cap. Crop runs before resize during local preprocessing.';
}
function samimateH3Duration(){let d=videoDuration('samimate')||0,start=timeVal('samimate_video_start',0),end=val('samimate_video_end')?timeVal('samimate_video_end',d):d;let span=end-start,cap=samimateWanFrameCap();return cap>0?Math.min(span||cap/24,cap/24):span}
function samimateH3Payload(){return {...h3Settings(),settings:payloadSettings(),generation_backend:'h3',pdd_enabled:false,source_workflow:'samimate',...samimateH3Options(),h3_endpoint_id:val('h3_endpoint_id'),task:'r2v',prompt_mode:val('samimate_prompt_mode'),prompt:val('samimate_prompt'),subject_prompt:val('samimate_subject_prompt'),duration:samimateH3Duration(),reference_paths:samimateReferencePaths(),image_path:val('samimate_image_path'),video_path:val('samimate_video_path'),reference_video_paths:[],reference_audio_paths:[],use_reference_audio_as_output:false,advanced_json:'{}',width:intFieldOrDefault('samimate_width',832),height:intFieldOrDefault('samimate_height',480),seed:Number(val('samimate_seed')),seed_random:Number(val('samimate_seed'))<0,filename_prefix:(val('h3_filename_prefix')||'RunPod_Media_Console_H3')+'_SAMimate'}}
async function runSamimate(){
  if(samimateGenerationBackend()==='h3')return sendSamimateH3();
  if(SAMIMATE_STATE?.active||$('samimate_run_button')?.disabled)return alert('A SAMimate run is already active.');
  let btn=$('samimate_run_button');
  if(btn)btn.disabled=true;
  try{
    await saveSettings();
    let video=val('samimate_video_path'), image=val('samimate_image_path');
    if(!video.trim())return alert('SAMimate needs a source video.');
    if(!image.trim())return alert('SAMimate needs a reference image.');
    if(chk('samimate_use_wan_points') && points.positive.length)applyPointsToPayload();
    let runFolder=await api('/api/samimate/run-folder',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    let generationBackend=samimateGenerationBackend();
    let generationPayload=generationBackend==='h3'?samimateH3Payload():null;
    let prepared=null;
    if(generationBackend==='h3'){
      if(btn){btn.disabled=true;btn.textContent='Preparing source...'}
      prepared=await api('/api/samimate/prepare-h3',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_path:video,video_start:val('samimate_video_start'),video_end:val('samimate_video_end'),video_crop:parseCrop('samimate'),video_frame_cap:val('samimate_video_frame_cap'),mask_frame_cap:samimateEffectiveFrameCap(),width:intFieldOrDefault('samimate_width',0),height:intFieldOrDefault('samimate_height',0)})});
      generationPayload={...generationPayload,...prepared,source_video_path:prepared.source_path};
    }
    SAMIMATE_STATE={active:true,prepared,startedAt:Date.now(),backend:generationBackend,sourceVideo:video,referenceImage:image,runDir:runFolder.path,runName:runFolder.name,trim:{start:val('samimate_video_start'),end:val('samimate_video_end'),frameCap:samimateEffectiveFrameCap(),crop:parseCrop('samimate')}};
    if(btn){btn.disabled=true;btn.textContent='Segmenting...'}
    clearOutputPreview('samimate');
    samimateSetStatus('Step 1/4: Segmenting source video with SAM3', `${samimateTrimSummary()}; output folder ${runFolder.name}`);
    let samPayloadData=samPayload();
    samPayloadData.source_path=video;
    samPayloadData.source_type='video';
    samPayloadData.frame_time=0;
    samPayloadData.video_start=val('samimate_video_start');
    samPayloadData.video_end=val('samimate_video_end');
    samPayloadData.video_crop=parseCrop('samimate');
    samPayloadData.preprocess_fps=val('samimate_fps');
    samPayloadData.fps=val('samimate_fps');
    samPayloadData.text_prompt=val('samimate_subject_prompt')||val('sam_text_prompt')||'person';
    samPayloadData.invert=false;
    samPayloadData.make_masked_video=true;
    samPayloadData.source_workflow='samimate';
    samPayloadData.output_dir=runFolder.path;
    samPayloadData.samimate_mask_frame_cap=samimateEffectiveFrameCap();
    samPayloadData.video_frame_cap=String(samPayloadData.samimate_mask_frame_cap||0);
    if(prepared){samPayloadData={...samPayloadData,text_prompt:generationPayload.subject_prompt||'person',source_path:prepared.source_path,video_start:0,video_end:'',video_crop:null,preprocess_fps:24,fps:24,samimate_mask_frame_cap:prepared.frame_count,video_frame_cap:String(prepared.frame_count)}}
    let stopSeg=startProgressLogger('samimate','SAMimate step 1/4 SAM3 segmentation',estimateSecFor('sam',{...samPayloadData,video_frame_cap:samPayloadData.video_frame_cap,fps:samPayloadData.fps}),'Segmenting with SAM3...');
    let samRes;
    try{
      samRes=await api('/api/sam/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(samPayloadData)});
      stopSeg('SAM3 segmentation complete');
    }catch(e){
      stopSeg('SAM3 segmentation failed');
      throw e;
    }
    SAM.result=samRes;samShowResult(samRes);
    if(!samRes.mask_video_path)throw new Error('SAM3 did not return a mask video.');
    if(!samRes.inverted_mask_video_path)throw new Error('SAM3 did not return an inverted mask video.');
    SAMIMATE_STATE={...SAMIMATE_STATE,subjectMask:samRes.mask_video_path,invertedMask:samRes.inverted_mask_video_path,maskedPreview:samRes.masked_video_path||''};
    samimateSetStatus('Step 2/4: SAM3 masks ready', `subject mask: ${samRes.mask_video_path}; background mask: ${samRes.inverted_mask_video_path}`);
    let backendLabel=generationBackend==='h3'?'MiniMax H3':'WanAnimate';
    samimateSetStatus(`Step 3/4: Submitting ${backendLabel}`,generationBackend==='h3'?`H3 masked source-latent replacement; ${samimateTrimSummary()}`:`subject mask is the edit area; ${samimateTrimSummary()}`);
    if(btn)btn.textContent=`Submitting ${generationBackend==='h3'?'H3':'Wan'}...`;
    let p=generationBackend==='h3'?{...generationPayload,mask_path:samRes.mask_video_path}:samimateWanPayload(samRes.mask_video_path);
    let runUrl=generationBackend==='h3'?'/api/run/h3':'/api/run/wananimate';
    let r=await api(runUrl,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    let job=r.job.id;log(`SAMimate ${backendLabel} job: `+job);log('SAMimate payload keys: '+r.payload_keys.join(', '));if(r.debug)log('SAMimate debug: '+JSON.stringify(r.debug));
    SAMIMATE_STATE={...SAMIMATE_STATE,job,endpoint:r.endpoint_id,submittedAt:Date.now()};
    samimateSetStatus(`Step 3/4: ${backendLabel} job submitted`, `job ${job}; polling every 4s`);
    watch(r.endpoint_id,job,'samimate',{payload:p,estimateSec:estimateSecFor('samimate',p),startedAt:SAMIMATE_STATE.submittedAt});
  }catch(e){if(SAMIMATE_STATE)SAMIMATE_STATE.active=false;samimateSetStatus('Error');log('SAMimate ERROR: '+e.message);alert('SAMimate error: '+e.message)}
  finally{if(btn){btn.disabled=false;btn.textContent='Run SAMimate'}}
}
async function submitRun(url,p,target='inf'){
  try{
    setOutputWaiting(target);
    let estimate=estimateSecFor(target,p),statusEl=targetStatusEl(target);
    if(statusEl)statusEl.textContent=`Submitting | ETA after start ~${fmtDurationSec(estimate)}`;
    log(`${targetLabel(target)}: submitting payload | estimated runtime ~${fmtDurationSec(estimate)}`);
    let r=await api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    let job=r.job.id;
    log(`${targetLabel(target)} submitted: ${job} | polling every 4s | ETA ~${fmtDurationSec(estimate)}`);
    log('Payload keys: '+r.payload_keys.join(', '));
    if(r.debug)log('Debug: '+JSON.stringify(r.debug));if(r.payload&&$(target+'_payload_preview'))$(target+'_payload_preview').value=JSON.stringify({endpoint_id:r.endpoint_id,payload:r.payload},null,2);
    watch(r.endpoint_id,job,target,{payload:p,estimateSec:estimate});
    return r;
  }catch(e){log('ERROR: '+e.message);throw e}
}
async function finishSamimate(items){try{let edited=(items||[]).find(x=>x.kind==='video'&&x.path),backendLabel=SAMIMATE_STATE?.backend==='h3'?'MiniMax H3':'WanAnimate';if(!edited)throw new Error(`${backendLabel} completed but no local video output was saved.`);samimateSetStatus(`Step 4/4: Compositing original background over ${backendLabel} result`, samimateTrimSummary());let stopComposite=startProgressLogger('samimate','SAMimate step 4/4 composite',Math.max(45,estimateSecFor('sam',{video_frame_cap:samimateWanFrameCap(),fps:+val('samimate_fps')||16})/3),'Compositing final video...');let r;try{r=await api('/api/samimate/composite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({foreground_path:edited.path,background_path:SAMIMATE_STATE.sourceVideo,mask_video_path:SAMIMATE_STATE.subjectMask,inverted_mask_video_path:SAMIMATE_STATE.invertedMask,run_dir:SAMIMATE_STATE.runDir,video_start:val('samimate_video_start'),video_end:val('samimate_video_end'),video_frame_cap:String(samimateWanFrameCap()),video_crop:parseCrop('samimate'),width:intFieldOrDefault('samimate_width',0),height:intFieldOrDefault('samimate_height',0),fps:+val('samimate_fps')||0,...(SAMIMATE_STATE.prepared?{generation_backend:'h3',background_path:SAMIMATE_STATE.prepared.composite_source_path||SAMIMATE_STATE.prepared.source_path,video_start:0,video_end:'',video_crop:null,video_frame_cap:String(SAMIMATE_STATE.prepared.frame_count),fps:24}: {})})});stopComposite('composite complete')}catch(e){stopComposite('composite failed');throw e}let out=r.items?.length?r.items:[r];showOutputPreview('samimate',out,true,{job:SAMIMATE_STATE.job,title:'SAMimate Run',note:`Saved ${out.length} outputs in ${r.run_dir||SAMIMATE_STATE.runDir}`});SAMIMATE_STATE.active=false;samimateSetStatus('Complete',`saved ${out.length} outputs in ${r.run_dir||SAMIMATE_STATE.runDir}`);if(r.debug)log('SAMimate composite debug: '+JSON.stringify(r.debug));log('SAMimate outputs saved: '+(r.run_dir||SAMIMATE_STATE.runDir))}catch(e){if(SAMIMATE_STATE)SAMIMATE_STATE.active=false;samimateSetStatus('Composite error');log('SAMimate composite error: '+e.message);alert('SAMimate composite error: '+e.message)}}
function watch(endpoint,job,target='inf',meta={}){
  let timerKey=`${endpoint}:${job}`;
  let pollStart=meta.startedAt||Date.now(),lastStatus='',lastProgressLog=0,estimate=meta.estimateSec||estimateSecFor(target,meta.payload||{}),statusEl=targetStatusEl(target);
  async function poll(){
    try{
      let r=await api(`/api/job/${endpoint}/${job}`),s=r.status.status,summary=progressSummary(target,s,pollStart,estimate,r),now=Date.now();
      if(statusEl)statusEl.textContent=summary;
      if(s!==lastStatus||now-lastProgressLog>30000||target==='samimate'){
        log(`${targetLabel(target)} ${job}: ${summary}`);
        lastStatus=s;lastProgressLog=now;
      }
      if(target==='samimate'&&s!=='COMPLETED'&&s!=='FAILED')samimateSetStatus(`Step 3/4: ${SAMIMATE_STATE?.backend==='h3'?'MiniMax H3':'WanAnimate'} ${s}`, summary);
      if(s==='COMPLETED'||s==='FAILED'||(target==='samimate'&&['CANCELLED','TIMED_OUT'].includes(s))){
        clearInterval(JOB_TIMERS[timerKey]);delete JOB_TIMERS[timerKey];
        let totalSec=(Date.now()-pollStart)/1000;
        if(s==='COMPLETED')rememberProgressDuration(target,totalSec);
        if(statusEl)statusEl.textContent=s==='COMPLETED'?`Complete | elapsed ${fmtDurationSec(totalSec)}`:`Failed | elapsed ${fmtDurationSec(totalSec)}`;
        let items=r.saved_items||[];
        if(r.saved?.length)log('Saved: '+r.saved.join(', '));
        if(r.uploaded_s3?.length)log('Uploaded to S3: '+r.uploaded_s3.join(', '));
        (r.download_errors||[]).forEach(e=>log('Download error: '+e));
        if(items.length){
          log('Preview items: '+items.map(x=>x.path||x.url).join(', '));
          if(target==='samimate'&&s==='COMPLETED'&&SAMIMATE_STATE){await finishSamimate(items)}
          else{showOutputPreview(target,items,true,{job,title:target==='inf'?'InfiniteTalk Run':(target==='h3'?'MiniMax H3 Run':'WanAnimate Run'),note:summary})}
          await refreshRecentOutputs(false);
        }else if(s==='COMPLETED'){
          if(target==='samimate'&&SAMIMATE_STATE){SAMIMATE_STATE.active=false;samimateSetStatus('Generation returned no video', 'Check endpoint errors in the job log; no composite was created.');}
          if(r.output_summary)log('No previewable outputs. Output summary: '+JSON.stringify(r.output_summary));
          showOutputPreview(target,[],true,{job,note:summary});
          await refreshRecentOutputs(false);
        }
        if(s==='FAILED'||(target==='samimate'&&['CANCELLED','TIMED_OUT'].includes(s))){
          setOutputIdle(target);
          if(target==='samimate'){if(SAMIMATE_STATE)SAMIMATE_STATE.active=false;}if(target==='samimate')samimateSetStatus(`${SAMIMATE_STATE?.backend==='h3'?'MiniMax H3':'WanAnimate'} failed`, summary);
          log(JSON.stringify(r.status,null,2));
          if(r.parsed_error)log('Parsed endpoint error: '+JSON.stringify(r.parsed_error,null,2));
          (r.error_hints||[]).forEach(h=>log('Hint: '+h));
          let err=(r.status&&r.status.error?String(r.status.error):'')+(r.parsed_error?JSON.stringify(r.parsed_error):'');
          if(!r.error_hints?.length)await explainAndLogError(err);
          if(err.includes('HTTP Error 400')&&err.includes('queue_prompt'))log('Hint: Comfy rejected the workflow before generation. If debug shows image_path/video_path, that means RunPod Volume Path delivery. Try Wan Delivery = Base64 / direct upload once to isolate file visibility. If Base64 also fails, try Mode=animate or disable Control Points to isolate the workflow.');
        }
        await loadJobHistory();
      }
    }catch(e){log('watch error: '+e.message)}
  }
  JOB_TIMERS[timerKey]=setInterval(poll,4000);
  poll();
}
function mediaKindFromPath(p){let ext=String(p||'').split('.').pop().toLowerCase();if(['png','jpg','jpeg','webp','bmp'].includes(ext))return'image';if(['mp4','mov','mkv','avi','webm'].includes(ext))return'video';if(['wav','mp3','m4a','flac','ogg'].includes(ext))return'audio';return'file'}
function clearOutputPreview(target){localStorage.removeItem(outputPreviewKey(target));let grid=$(target+'_output_grid'),empty=$(target+'_output_empty');if(grid)grid.innerHTML='';if(empty){empty.textContent='Waiting for completed output...';empty.classList.remove('hidden')}}
function setOutputWaiting(target){let empty=$(target+'_output_empty');if(empty&&!loadOutputRuns(target).length){empty.textContent='Waiting for completed output...';empty.classList.remove('hidden')}}
function setOutputIdle(target){let empty=$(target+'_output_empty');if(empty&&!loadOutputRuns(target).length){empty.textContent='Completed outputs will appear here.';empty.classList.remove('hidden')}}
function normalizeOutputRuns(raw){if(!Array.isArray(raw))return[];if(!raw.length)return[];if(raw[0]&&Array.isArray(raw[0].items))return raw;return raw.length?[{id:'legacy_'+Date.now(),createdAt:Date.now(),title:'Previous Output',items:raw}]:[]}
function outputRunLimit(target){return 24}
function loadOutputRuns(target){try{let runs=normalizeOutputRuns(JSON.parse(localStorage.getItem(outputPreviewKey(target))||'[]')),limit=outputRunLimit(target);if(runs.length>limit){runs=runs.slice(0,limit);localStorage.setItem(outputPreviewKey(target),JSON.stringify(runs))}return runs}catch{return[]}}
function saveOutputRuns(target,runs){try{localStorage.setItem(outputPreviewKey(target),JSON.stringify((runs||[]).slice(0,outputRunLimit(target))))}catch{}}
function runTitle(target,n){return target==='inf'?`InfiniteTalk Run ${n}`:target==='wan'?`WanAnimate Run ${n}`:target==='h3'?`MiniMax H3 Run ${n}`:target==='samimate'?`SAMimate Run ${n}`:`Run ${n}`}
function showOutputPreview(target,items,persist=true,meta={}){let runs=loadOutputRuns(target);if(persist){if(meta.job)runs=runs.filter(run=>run.job!==meta.job);runs.unshift({id:`${Date.now()}_${Math.random().toString(16).slice(2)}`,createdAt:Date.now(),job:meta.job||'',title:meta.title||runTitle(target,runs.length+1),note:meta.note||'',items:items||[]});runs=runs.slice(0,outputRunLimit(target));saveOutputRuns(target,runs)}else if(!runs.length&&items?.length){runs=normalizeOutputRuns(items).slice(0,outputRunLimit(target))}renderOutputRuns(target,runs)}
function renderOutputRuns(target,runs=loadOutputRuns(target)){let grid=$(target+'_output_grid'),empty=$(target+'_output_empty');if(!grid)return;grid.innerHTML='';if(!runs.length){if(empty){empty.textContent='Completed outputs will appear here.';empty.classList.remove('hidden')}return}if(empty)empty.classList.add('hidden');runs.forEach((run,idx)=>{let frame=document.createElement('section');frame.className='outputRunFrame';let count=runs.length-idx,when=run.createdAt?new Date(run.createdAt).toLocaleString():'';let sub=[when,run.job?`job ${run.job}`:'',run.note||''].filter(Boolean).join(' • ');let items=run.items||[];frame.innerHTML=`<div class="outputRunHeader"><div><h4>${escapeHtml(run.title||runTitle(target,count))}</h4><small>${escapeHtml(sub)}</small></div><span>${items.length} item${items.length===1?'':'s'}</span></div><div class="outputRunItems"></div>`;let body=frame.querySelector('.outputRunItems');if(!items.length){let msg=document.createElement('div');msg.className='outputEmpty inline';msg.textContent='No previewable outputs were returned for this run.';body.appendChild(msg)}else{for(const it of items){body.appendChild(outputItemElement(it))}}grid.appendChild(frame)})}
function outputItemElement(it){let card=document.createElement('div');card.className='outputItem';let url=cacheBustUrl(it.url||('/api/file?path='+encodeQS(it.path)));let media='';if(it.kind==='video')media=`<video controls muted preload="metadata" src="${url}"></video>`;else if(it.kind==='image')media=`<img src="${url}" alt="Output preview">`;else if(it.kind==='audio')media=`<audio controls preload="metadata" src="${url}"></audio>`;else media=`<div class="outputFile">File</div>`;let label=it.label?`<div class="outputLabel">${escapeHtml(it.label)}</div>`:'';card.innerHTML=`${label}${media}<input readonly value="${escapeHtml(it.path||'')}">`;return card}
function togglePointsEditor(){let body=$('points_editor_body'),btn=document.querySelector('[onclick="togglePointsEditor()"]');if(!body)return;let hidden=body.classList.toggle('hidden');if(btn)btn.textContent=hidden?'Show editor':'Hide editor'}
function encodeQS(v){return encodeURIComponent(v||'')}
function cacheBustUrl(url){if(!url)return '';return url+(url.includes('?')?'&':'?')+'t='+Date.now()}
function isHttpPath(p){return /^https?:\/\//i.test(String(p||''))}
function isS3Path(p){return /^s3:\/\//i.test(String(p||''))}
function s3KeyFromUri(p){let m=String(p||'').match(/^s3:\/\/[^/]+\/(.+)$/i);return m?m[1]:String(p||'')}
function s3StorageForPath(path){let value=String(path||''),bucket=value.startsWith('s3://')?value.slice(5).split('/',1)[0]:'';let h3=String(SETTINGS.h3_storage?.s3_bucket||SETTINGS.h3_storage?.network_volume_id||''),main=String(SETTINGS.s3_bucket||'');if(bucket&&h3&&bucket===h3)return'h3';if(bucket&&main&&bucket===main)return'main';return val('browser_storage')||BROWSER_STORAGE||SETTINGS.browser_s3_storage||'h3'}
async function previewUrlForPath(path){path=String(path||'').trim();if(!path)return'';if(isHttpPath(path))return path;if(isS3Path(path)){let storage=s3StorageForPath(path),d=await api('/api/s3/details?key='+encodeQS(s3KeyFromUri(path))+'&storage='+encodeQS(storage));return d.file_url||''}return '/api/file?path='+encodeQS(path)}
async function persistSettingsQuietly(){try{SETTINGS=await api('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(gatherSettings())})}catch(e){log('Settings persist failed: '+e.message)}}
async function restoreInputPreview(id){let p=val(id).trim();if(!p)return;try{let url=await previewUrlForPath(p);if(url)previewForInput(id,url)}catch(e){log('Preview restore failed for '+id+': '+e.message)}}
async function restoreSamOutputs(){SAM.result={mask_path:val('sam_mask_path'),mask_video_path:val('sam_mask_video_path'),overlay_path:val('sam_overlay_path'),cutout_path:val('sam_cutout_path'),masked_image_path:val('sam_masked_image_path'),masked_video_path:val('sam_masked_video_path')};for(const [id,kind] of [['sam_mask_video','video'],['sam_masked_video','video']]){let p=val(id+'_path').trim(),el=$(id+'_preview');if(!p||!el)continue;try{let url=cacheBustUrl(await previewUrlForPath(p));el.src=url;el.load();el.classList.remove('hidden')}catch(e){log('SAM preview restore failed for '+id+': '+e.message)}}}
async function restoreTtsPreviews(){for(const [pathId,audioId] of [['tts_output_path','tts_preview'],['inf_tts_output_path','inf_tts_preview']]){let p=val(pathId).trim(),a=$(audioId);if(!p||!a)continue;try{a.src=cacheBustUrl(await previewUrlForPath(p));a.classList.remove('hidden');a.load()}catch(e){log('TTS preview restore failed for '+pathId+': '+e.message)}}}
function outputPreviewKey(target){return 'output_preview_'+target}
function restoreStoredOutputPreview(target){try{let raw=localStorage.getItem(outputPreviewKey(target));if(raw)showOutputPreview(target,JSON.parse(raw),false)}catch(e){localStorage.removeItem(outputPreviewKey(target))}}
function queueTabPreviewRestore(tab,force=false){PREVIEW_RESTORE_QUEUE=PREVIEW_RESTORE_QUEUE.then(()=>restoreTabPreviews(tab,force)).catch(e=>log('Preview restore failed: '+e.message));return PREVIEW_RESTORE_QUEUE}
async function restoreTabPreviews(tab,force=false){
  if(!tab||(!force&&RESTORED_TABS.has(tab)))return;
  const ids={inf:['inf_image_path','inf_video_path','inf_audio_path','inf_audio2_path'],wan:['wan_image_path','wan_video_path','wan_masked_video_path'],h3:['h3_first_frame_path','h3_last_frame_path',...Array.from({length:9},(_,i)=>`h3_reference${i+1}_path`),...Array.from({length:3},(_,i)=>`h3_reference_video${i+1}_path`),...Array.from({length:3},(_,i)=>`h3_reference_audio${i+1}_path`)],samimate:['samimate_image_path','samimate_video_path',...Array.from({length:8},(_,i)=>`samimate_reference${i+2}_path`)],sam:['sam_source_path']}[tab]||[];
  RESTORING_PREVIEWS=true;
  try{
    for(const id of ids)await restoreInputPreview(id);
    if(tab==='sam'){if(val('sam_source_path').trim())await samLoadFrame();await restoreSamOutputs();samDraw()}
    if(tab==='tts')await restoreTtsPreviews();
    if(['inf','wan','h3','samimate','sam','tts','image_editor'].includes(tab))restoreStoredOutputPreview(tab);
    if(tab==='inf'){updateInfiniteModeUI();updateTimeline('inf');updateAudioTimeline('inf_audio');updateAudioTimeline('inf_audio2')}
    if(tab==='wan')updateTimeline('wan');
    if(tab==='h3')updateH3TaskUI();
    if(tab==='samimate'){updateTimeline('samimate');samimateRefreshModelOptions()}
    RESTORED_TABS.add(tab);
  }finally{RESTORING_PREVIEWS=false}
}
function browserIcon(kind){return kind==='folder'?'📁':kind==='image'?'🖼️':kind==='video'?'🎞️':kind==='audio'?'🎧':'📄'}
let BROWSER_SELECTED='';
function browserKindForTarget(id,kind){if(id==='wan_mask_path')return'All';if(kind==='image'||id?.includes('image')||id?.includes('mask'))return'Images';if(kind==='video'||id?.includes('video'))return'Videos';if(kind==='audio'||id?.includes('audio'))return'Audio';return'All'}
function hidePathRecent(){document.querySelector('.pathRecentMenu')?.remove()}
function inputRecentKey(id){return 'recent_paths_'+id}
function recentPathsForInput(id){try{return JSON.parse(localStorage.getItem(inputRecentKey(id))||'[]')}catch{return[]}}
function rememberPathForInput(id,path){if(!id||!path||String(path).startsWith('http'))return;let items=recentPathsForInput(id).filter(x=>x!==path);items.unshift(path);localStorage.setItem(inputRecentKey(id),JSON.stringify(items.slice(0,8)))}
function togglePathRecent(id,event){event?.preventDefault();event?.stopPropagation();let input=$(id);if(!input)return;let open=document.querySelector('.pathRecentMenu');if(open&&open.dataset.inputId===id){hidePathRecent();return}showPathRecent(input,true)}
function showPathRecent(input,showAll=false){let needle=showAll?'':input.value.trim().toLowerCase(),items=recentPathsForInput(input.id).filter(p=>!needle||p.toLowerCase().includes(needle));hidePathRecent();if(!items.length)return;let box=input.getBoundingClientRect(),menu=document.createElement('div');menu.className='pathRecentMenu';menu.dataset.inputId=input.id;menu.style.left=box.left+'px';menu.style.top=(box.bottom+4)+'px';menu.style.width=Math.max(box.width,360)+'px';menu.innerHTML=items.map(p=>`<button type="button" data-path="${escapeHtml(p)}"><span>${browserIcon(mediaKindFromPath(p))}</span><div><strong>${escapeHtml(p.split(/[\\/]/).pop()||p)}</strong><small>${escapeHtml(p)}</small></div></button>`).join('');menu.addEventListener('mousedown',e=>e.preventDefault());menu.addEventListener('click',async e=>{let b=e.target.closest('button[data-path]');if(!b)return;if(input.id==='wan_mask_path')setWanMaskPretrimmed(false);set(input.id,b.dataset.path);previewForInput(input.id,await previewUrlForPath(b.dataset.path));await persistSettingsQuietly();hidePathRecent()});document.body.appendChild(menu)}
function clearRecentMedia(){Object.keys(localStorage).filter(k=>k.startsWith('recent_paths_')).forEach(k=>localStorage.removeItem(k));hidePathRecent();log('Path history cleared')}
function escapeHtml(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]))}
function jsString(s){return JSON.stringify(String(s??''))}
function hashString(s){let h=0;for(let i=0;i<s.length;i++){h=((h<<5)-h)+s.charCodeAt(i);h|=0}return Math.abs(h)}
async function loadBrowserFavorites(){try{const r=await api('/api/favorites');const sel=$('browser_favorites');if(!sel)return;sel.innerHTML='<option value="">Favorites</option>'+(r.favorites||[]).map(p=>`<option value="${escapeHtml(p)}">${escapeHtml(p.split(/[\\/]/).pop()||p)}</option>`).join('')}catch(e){log('favorites error: '+e.message)}}
function browserFileMeta(it){if(it.is_dir)return (it.source==='s3'?'S3 folder':'Folder');let bits=[it.source==='s3'?'S3':it.ext||it.kind];if(it.size_mb!==null&&it.size_mb!==undefined)bits.push(`${it.size_mb} MB`);if(it.modified)bits.push(String(it.modified).slice(0,16));return bits.join(' • ')}
function browserSizeText(it){if(it.is_dir)return'';if(Number.isFinite(Number(it.size)))return Number(it.size).toLocaleString();if(it.size_mb!==null&&it.size_mb!==undefined)return `${it.size_mb} MB`;return''}
function browserDateText(v){if(!v)return'';let s=String(v).replace('T',' ');return s.slice(0,16)}
function browserCommentText(it){if(it.is_dir)return it.source==='s3'?'S3 folder':'Folder';let parts=[];if(it.source==='s3')parts.push('S3');parts.push((it.ext||it.kind||'file').replace('.',''));return parts.filter(Boolean).join(' / ')}
function setBrowserSort(col){let cur=val('browser_sort_by')||'name',dir=val('browser_sort_dir')||'asc';set('browser_sort_by',col);set('browser_sort_dir',cur===col&&dir==='asc'?'desc':'asc');browse()}
function sortBrowserFoldersByModified(){set('browser_sort_by','modified');set('browser_sort_dir','desc');browse()}
function browserSortMark(col){return (val('browser_sort_by')||'name')===col?((val('browser_sort_dir')||'asc')==='asc'?' ▲':' ▼'):''}
function updateBrowserSourceUI(){let s=val('browser_source')||'local',lab=$('browse_path_label'),inp=$('browse_path');if(lab)lab.childNodes[0].nodeValue=s==='s3'?'S3 Prefix':'Folder';if(inp)inp.placeholder=s==='s3'?'Blank for bucket root, output/h3, or models/loras/H3':'Folder path';$('browser_storage_label')?.classList.toggle('hidden',s!=='s3')}
async function browse(path){
  if(path!==undefined)$('browse_path').value=path;
  BROWSER_SOURCE=val('browser_source')||'local';
  BROWSER_STORAGE=val('browser_storage')||SETTINGS.browser_s3_storage||'h3';
  updateBrowserSourceUI();
  const qs=new URLSearchParams({media_type:val('browser_type')||'All',search:val('browser_search')||'',recursive:chk('browser_recursive')?'true':'false',sort_by:val('browser_sort_by')||'name',sort_dir:val('browser_sort_dir')||'asc',_t:String(Date.now())});
  let r;
  try{
    if(BROWSER_SOURCE==='s3'){
      qs.set('prefix',$('browse_path').value||'');
      qs.set('storage',BROWSER_STORAGE);
      r=await api('/api/s3/browse?'+qs.toString());
    }else{
      qs.set('path',$('browse_path').value||'');
      r=await api('/api/browse?'+qs.toString());
    }
  }catch(e){
    if(BROWSER_SOURCE==='s3'){
      log('S3 browse retry: '+e.message);
      await new Promise(resolve=>setTimeout(resolve,700));
      r=await api('/api/s3/browse?'+qs.toString());
    }else{
      throw e;
    }
  }
  CURRENT_BROWSER=r.path;
  $('browse_path').value=r.path;
  clearBrowserSelection();
  BROWSER_VISIBLE_ITEMS=(r.items||[]).filter(it=>!it.is_dir).map(it=>({path:it.path,key:it.key||'',source:it.source||BROWSER_SOURCE,storage:it.storage||BROWSER_STORAGE,kind:it.kind||mediaKindFromPath(it.path),name:it.name||''}));
  let parentAction=BROWSER_SOURCE==='s3'?`browse(${jsString(r.parent||'')})`:`browse(${jsString(r.parent)})`;
  let html=`<div class="browserTable"><div class="browserHeader"><button onclick="setBrowserSort('name')" title="Sort by file or folder name">Name${browserSortMark('name')}</button><button onclick="setBrowserSort('size')" title="Sort by file size">Size${browserSortMark('size')}</button><button onclick="setBrowserSort('modified')" title="Sort by last modified time">Modified${browserSortMark('modified')}</button><button onclick="setBrowserSort('created')" title="Sort by creation time when available">Created${browserSortMark('created')}</button><button onclick="setBrowserSort('type')" title="Sort by media type or source">Comment${browserSortMark('type')}</button></div><div class="browserRow browserParentRow" onclick="${escapeHtml(parentAction)}" title="Open the parent ${BROWSER_SOURCE==='s3'?'prefix':'folder'}"><div class="browserNameCell"><span class="browserFileIcon">📁</span><span>..</span></div><div></div><div>Parent ${BROWSER_SOURCE==='s3'?'prefix':'folder'}</div><div></div><div></div></div>`;
  for(const it of r.items){
    const p=it.path,icon=browserIcon(it.kind);
    const action=it.is_dir?`browse(${jsString(it.key||p)})`:`selectBrowserFile(event,${jsString(p)},${jsString(it.key||'')},${jsString(it.source||BROWSER_SOURCE)},${jsString(it.storage||BROWSER_STORAGE)})`;
    html+=`<div class="browserRow ${it.source==='s3'?'s3Row':''}" id="browser_row_${hashString(p)}" onclick="${escapeHtml(action)}" title="${escapeHtml(it.is_dir?'Open folder/prefix: '+p:'Select and preview: '+p)}"><div class="browserNameCell"><span class="browserFileIcon">${icon}</span><span class="browserName">${escapeHtml(it.name)}</span></div><div class="browserSizeCell">${escapeHtml(browserSizeText(it))}</div><div>${escapeHtml(browserDateText(it.modified))}</div><div>${escapeHtml(browserDateText(it.created))}</div><div>${escapeHtml(browserCommentText(it))}</div></div>`;
  }
  html+='</div>';
  $('browser_list').innerHTML=html;
  applyTooltips($('browser_list'));
  if(r.is_truncated)log('S3 listing truncated at 500 items. Use search or a deeper prefix.');
  if(r.folder_modified_truncated)log('Folder modified-date scan reached 5,000 objects; use a deeper prefix for an exact folder order.');
}
function browseHome(){if((val('browser_source')||'local')==='s3')browse('');else browse('')}function browseParent(){if((val('browser_source')||'local')==='s3'){let p=CURRENT_BROWSER.replace(/\/$/,'').split('/').slice(0,-1).join('/');browse(p?p+'/':'')}else browse(CURRENT_BROWSER.split(/[\\/]/).slice(0,-1).join('/'))}
function clearBrowserSelection(){BROWSER_SELECTED='';BROWSER_SELECTED_ITEM=null;BROWSER_SELECTED_ITEMS={};BROWSER_LAST_INDEX=-1;updateBrowserSelectionClasses()}
function selectedBrowserItems(){return Object.values(BROWSER_SELECTED_ITEMS)}
function setBrowserSelectedItem(item,on=true){if(on)BROWSER_SELECTED_ITEMS[item.path]=item;else delete BROWSER_SELECTED_ITEMS[item.path]}
function updateBrowserSelectionClasses(){document.querySelectorAll('.browserRow').forEach(x=>x.classList.remove('selected'));for(const item of selectedBrowserItems()){const row=$('browser_row_'+hashString(item.path));if(row)row.classList.add('selected')}}
function browserSelectionSummary(){let items=selectedBrowserItems();return items.length>1?`\n\n${items.length} files selected.`:''}
async function selectBrowserFile(e,p,key='',source='local',storage=''){storage=storage||BROWSER_STORAGE;let idx=BROWSER_VISIBLE_ITEMS.findIndex(x=>x.path===p),clicked=idx>=0?BROWSER_VISIBLE_ITEMS[idx]:{path:p,key,source,storage,kind:mediaKindFromPath(p)};if(e?.shiftKey&&BROWSER_LAST_INDEX>=0&&idx>=0){let [a,b]=[BROWSER_LAST_INDEX,idx].sort((x,y)=>x-y);if(!e.ctrlKey&&!e.metaKey)BROWSER_SELECTED_ITEMS={};for(const item of BROWSER_VISIBLE_ITEMS.slice(a,b+1))setBrowserSelectedItem(item,true)}else if(e?.ctrlKey||e?.metaKey){setBrowserSelectedItem(clicked,!BROWSER_SELECTED_ITEMS[p]);if(BROWSER_LAST_INDEX<0&&idx>=0)BROWSER_LAST_INDEX=idx}else{BROWSER_SELECTED_ITEMS={};setBrowserSelectedItem(clicked,true)}if(idx>=0)BROWSER_LAST_INDEX=idx;BROWSER_SELECTED=p;BROWSER_SELECTED_ITEM=clicked;updateBrowserSelectionClasses();let d=source==='s3'?await api('/api/s3/details?key='+encodeQS(key)+'&storage='+encodeQS(storage)):await api('/api/media/details?path='+encodeQS(p));$('browser_preview_empty').classList.add('hidden');for(const id of ['browser_image_preview','browser_video_preview','browser_audio_preview']){let el=$(id);el.classList.add('hidden');el.removeAttribute('src');if(el.load)el.load()}let fileUrl=d.file_url?cacheBustUrl(d.file_url):'';if(source==='s3'&&fileUrl)log(`Caching ${storage.toUpperCase()} S3 preview locally: `+d.key);if(d.kind==='image'){$('browser_image_preview').src=fileUrl;$('browser_image_preview').classList.remove('hidden')}else if(d.kind==='video'){$('browser_video_preview').src=fileUrl;$('browser_video_preview').classList.remove('hidden');$('browser_video_preview').load()}else if(d.kind==='audio'){$('browser_audio_preview').src=fileUrl;$('browser_audio_preview').classList.remove('hidden');$('browser_audio_preview').load()}else{$('browser_preview_empty').textContent=d.name;$('browser_preview_empty').classList.remove('hidden')}$('browser_details').textContent=Object.entries(d).filter(([k])=>!['file_url','thumb_url'].includes(k)).map(([k,v])=>`${k}: ${v}`).join('\n')+browserSelectionSummary();navigator.clipboard?.writeText(p).catch(()=>{})}
async function setBrowserTarget(id=BROWSER_TARGET){if(!BROWSER_SELECTED)return alert('Select a file first.');if(!id)return alert('Choose a target field first.');let previewUrl=BROWSER_SELECTED_ITEM?.source==='s3'?await previewUrlForPath(BROWSER_SELECTED):'/api/file?path='+encodeQS(BROWSER_SELECTED);if(id==='wan_mask_path')setWanMaskPretrimmed(false);set(id,BROWSER_SELECTED);previewForInput(id,previewUrl);await persistSettingsQuietly();log('Set '+id+' = '+BROWSER_SELECTED)}
async function sendBrowserSelectionToH3(kind){let items=selectedBrowserItems();if(!items.length&&BROWSER_SELECTED_ITEM)items=[BROWSER_SELECTED_ITEM];items=items.filter(item=>(item.kind||mediaKindFromPath(item.path))===kind);if(!items.length)return alert(`Select at least one ${kind} file first.`);let max=kind==='image'?9:3,stem=kind==='image'?'h3_reference':`h3_reference_${kind}`,slots=[];for(let i=1;i<=max;i++)if(!val(`${stem}${i}_path`).trim())slots.push(i);if(!slots.length)return alert(`All H3 ${kind} reference slots are already filled.`);let added=0;for(const item of items.slice(0,slots.length)){let id=`${stem}${slots[added]}_path`;set(id,item.path);try{previewForInput(id,await previewUrlForPath(item.path))}catch(e){log(`H3 ${kind} preview failed: `+e.message)}added++}set('h3_task','r2v');syncH3SubjectDefinitions();updateH3ReferenceLabels();updateH3TaskUI();await persistSettingsQuietly();log(`Sent ${added} ${kind} reference${added===1?'':'s'} from Media Browser to MiniMax H3.`)}
function browseForTarget(id,kind='media'){BROWSER_TARGET=id;let btn=$('browser_use_target');if(btn){btn.textContent='Use selected for '+id.replace(/_/g,' ');btn.classList.remove('hidden')}set('browser_type',browserKindForTarget(id,kind));showTab('browser');browse(val('browse_path')||SETTINGS.output_dir||'')}
async function downloadS3Selected(){let items=selectedBrowserItems().filter(x=>x.source==='s3'&&x.key);if(!items.length&&BROWSER_SELECTED_ITEM?.source==='s3')items=[BROWSER_SELECTED_ITEM];if(!items.length)return alert('Select one or more S3 objects first.');let storage=items[0].storage||BROWSER_STORAGE,r=await api('/api/s3/download-many',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys:items.map(x=>x.key),storage})});log(`Downloaded ${r.files} ${storage.toUpperCase()} S3 object(s), ${r.mb} MB -> ${r.path}`);if(r.items?.length){let first=r.items[0];BROWSER_SELECTED=first.path;BROWSER_SELECTED_ITEM={path:first.path,key:'',source:'local'};BROWSER_SELECTED_ITEMS={[first.path]:BROWSER_SELECTED_ITEM};await selectBrowserFile(null,first.path,'','local')}alert(`Downloaded ${r.files} file(s) to ${r.path}`)}
async function downloadS3CurrentPrefix(){if((val('browser_source')||'local')!=='s3')return alert('Switch Source to S3 Bucket first.');let prefix=CURRENT_BROWSER||val('browse_path')||'',storage=val('browser_storage')||BROWSER_STORAGE;if(!confirm(`Download every object from the ${storage.toUpperCase()} bucket under this prefix?\\n\\n${prefix||'(bucket root)'}`))return;let r=await api('/api/s3/download-prefix',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prefix,storage})});log(`Downloaded ${storage.toUpperCase()} S3 prefix ${prefix||'(root)'}: ${r.files} files, ${r.mb} MB -> ${r.path}`);alert(`Downloaded ${r.files} files to ${r.path}`)}
function addBrowserFileToPrompt(){if(!BROWSER_SELECTED)return alert('Select a file first.');const text=BROWSER_SELECTED.split(/[\\/]/).pop().replace(/\.[^.]+$/,'').replace(/[_-]+/g,' ');const active=document.querySelector('.tab.active');const target=active?.id==='inf'?'inf_prompt':active?.id==='wan'?'wan_prompt':'wan_prompt';set(target,(val(target)+' '+text).trim());log('Added to prompt: '+text)}
async function addBrowserFavorite(){await api('/api/favorites',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:CURRENT_BROWSER||val('browse_path')})});await loadBrowserFavorites();log('Favorite folder added: '+(CURRENT_BROWSER||val('browse_path')))}
async function clearBrowserFavorites(){if(!confirm('Clear all favorite folders?'))return;await api('/api/favorites',{method:'DELETE'});await loadBrowserFavorites();log('Browser favorites cleared')}
function setupBrowserEvents(){$('browser_source')?.addEventListener('change',()=>{clearBrowserSelection();updateBrowserSourceUI();browse((val('browser_source')||'local')==='s3'?'':SETTINGS.output_dir||'')});$('browser_storage')?.addEventListener('change',async()=>{BROWSER_STORAGE=val('browser_storage')||'h3';clearBrowserSelection();await persistSettingsQuietly();browse('')});$('browser_type')?.addEventListener('change',()=>browse());$('browser_sort_by')?.addEventListener('change',()=>browse());$('browser_sort_dir')?.addEventListener('change',()=>browse());$('browser_recursive')?.addEventListener('change',()=>browse());$('browser_search')?.addEventListener('input',()=>{clearTimeout(window.__browseSearchTimer);window.__browseSearchTimer=setTimeout(()=>browse(),180)});$('browser_favorites')?.addEventListener('change',e=>{if(e.target.value)browse(e.target.value)});$('browse_path')?.addEventListener('keydown',e=>{if(e.key==='Enter')browse()});updateBrowserSourceUI()}
function setupWorkflowSuite(){
  injectWorkflowTools('inf','InfiniteTalk');
  injectWorkflowTools('wan','WanAnimate');
  injectWorkflowTools('h3','MiniMax H3');
  injectWorkflowTools('samimate','SAMimate');
  injectSettingsProfiles();
  injectMediaRepairTools();
}
function injectWorkflowTools(tab,label){
  let section=$(tab);if(!section||$(tab+'_workflow_tools'))return;
  let card=document.createElement('div');card.id=tab+'_workflow_tools';card.className='card workflowTools';
  card.innerHTML=`<div class="cardTitleRow"><h3>${label} Workflow Tools</h3><div class="logActions"><button type="button" onclick="validateCurrent('${tab}')">Validate</button><button type="button" onclick="previewPayload('${tab}')">Preview payload</button></div></div>
  <div class="workflowToolGrid">
    <div><label>Preset<select id="${tab}_preset_select"><option value="">Preset</option></select></label><div class="buttonRow"><button type="button" onclick="saveCurrentPreset('${tab}')">Save preset</button><button type="button" onclick="applyPreset('${tab}')">Load preset</button><button type="button" onclick="deletePreset('${tab}')">Delete</button></div></div>
    <div><h4>Validation</h4><div id="${tab}_validation" class="validationList">Not checked yet.</div></div>
  </div>
  <details class="payloadDetails"><summary>Payload Inspector</summary><textarea id="${tab}_payload_preview" rows="10" readonly></textarea></details>`;
  let output=section.querySelector('.outputPanel');section.insertBefore(card,output||null);
}
function injectSettingsProfiles(){
  let settings=$('settings');if(!settings||$('endpoint_profiles_card'))return;
  let card=document.createElement('div');card.id='endpoint_profiles_card';card.className='card';
  card.innerHTML=`<div class="cardTitleRow"><h3>Endpoint Profiles</h3><div class="logActions"><button type="button" onclick="saveEndpointProfile()">Save current</button><button type="button" onclick="applyEndpointProfile()">Apply</button><button type="button" onclick="deleteEndpointProfile()">Delete</button></div></div><label>Profile<select id="endpoint_profile_select"><option value="">Profile</option></select></label><p class="hint">Profiles store endpoint IDs, delivery mode, and core S3/volume routing so you can switch between known-good endpoint setups.</p>`;
  settings.appendChild(card);
}
function injectMediaRepairTools(){
  let browser=$('browser');if(!browser||$('media_repair_card'))return;
  let card=document.createElement('div');card.id='media_repair_card';card.className='card mediaRepairCard';
  card.innerHTML=`<div class="cardTitleRow"><h3>Media Repair</h3><div class="logActions"><button type="button" onclick="repairSelectedMedia('browser_mp4')">Browser MP4</button><button type="button" onclick="repairSelectedMedia('normalize_fps')">Normalize FPS</button><button type="button" onclick="repairSelectedMedia('binary_mask')">Binary Mask</button><button type="button" onclick="repairSelectedMedia('resize')">Resize</button></div></div><div class="fields compact"><label>Resize W<input id="repair_width" type="number" placeholder="width"></label><label>Resize H<input id="repair_height" type="number" placeholder="height"></label><label>FPS<input id="repair_fps" type="number" placeholder="16"></label></div><div id="repair_status" class="validationList">Select a media file, then choose a repair action.</div>`;
  browser.appendChild(card);
}
function tabSettingsKey(tab){return tab==='inf'?'infinitetalk':tab==='wan'?'wananimate':tab==='h3'?'h3':'samimate'}
function currentPayload(tab){
  if(tab==='inf')return {settings:payloadSettings(),delivery_mode:val('delivery_mode'),input_type:val('inf_input_type'),person_count:val('inf_person_count'),image_path:val('inf_image_path'),video_path:val('inf_video_path'),audio_path:val('inf_audio_path'),audio2_path:val('inf_audio2_path'),audio_start:val('inf_audio_start'),audio_end:val('inf_audio_end'),audio2_start:val('inf_audio2_start'),audio2_end:val('inf_audio2_end'),video_start:val('inf_video_start'),video_end:val('inf_video_end'),video_frame_cap:val('inf_video_frame_cap'),video_crop:parseCrop('inf'),prompt:val('inf_prompt'),advanced_json:val('inf_advanced'),width:+val('inf_width'),height:+val('inf_height'),max_frame:val('inf_max_frame'),sync_to_audio:chk('inf_sync_to_audio'),force_offload:chk('inf_force_offload'),network_volume:chk('inf_network_volume')};
  if(tab==='wan'){let pointPayload=readPointPayloadFromFields();return {settings:payloadSettings(),delivery_mode:val('delivery_mode'),wan_delivery_override:val('wan_delivery_override'),image_path:val('wan_image_path'),video_path:val('wan_video_path'),mask_path:val('wan_mask_path'),masked_video_path:val('wan_masked_video_path'),mask_pretrimmed:wanMaskPretrimmed(),video_start:val('wan_video_start'),video_end:val('wan_video_end'),video_frame_cap:val('wan_video_frame_cap'),video_crop:parseCrop('wan'),prompt:val('wan_prompt'),negative_prompt:val('wan_negative_prompt'),advanced_json:val('wan_advanced'),mode:val('wan_mode'),width:intFieldOrDefault('wan_width',0),height:intFieldOrDefault('wan_height',0),seed:+val('wan_seed')||12345,fps:+val('wan_fps')||16,cfg:+val('wan_cfg')||1,steps:+val('wan_steps')||6,pose_estimation:chk('wan_pose'),face_detection:chk('wan_face'),mask_editing:chk('wan_mask_edit'),network_volume:chk('wan_network_volume'),control_points_enabled:chk('wan_control_points'),...pointPayload}};
  if(tab==='h3')return {settings:payloadSettings(),h3_endpoint_id:val('h3_endpoint_id'),...h3Settings()};
  return samimateGenerationBackend()==='h3'?samimatePreparedPayload():samimateWanPayload(val('sam_mask_video_path')||'');
}
function renderValidation(tab,r){
  let el=$(tab+'_validation')||$('workflow_status');if(!el)return;
  let parts=[];(r.errors||[]).forEach(x=>parts.push(`<div class="validationItem bad">${escapeHtml(x)}</div>`));(r.warnings||[]).forEach(x=>parts.push(`<div class="validationItem warn">${escapeHtml(x)}</div>`));if(!parts.length)parts.push('<div class="validationItem good">Ready. No blocking validation issues found.</div>');
  if(r.info?.resolved_size)parts.push(`<div class="validationItem">Resolved size: ${r.info.resolved_size.width} x ${r.info.resolved_size.height}</div>`);
  if(r.info?.effective_video_duration&&r.info?.effective_mask_duration)parts.push(`<div class="validationItem">Effective trim length: video ${r.info.effective_video_duration.effective}s / mask ${r.info.effective_mask_duration.effective}s</div>`);
  if(r.info?.mask_duration_note)parts.push(`<div class="validationItem good">${escapeHtml(r.info.mask_duration_note)}</div>`);
  el.innerHTML=parts.join('');
}
async function validateCurrent(tab){try{let r=await api('/api/validate/'+tab,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentPayload(tab))});renderValidation(tab,r);if(tab==='samimate'&&$('workflow_status'))renderValidation('workflow',r);log(`${tab} validation: ${r.ok?'ok':'needs attention'}`);return r}catch(e){log('Validation error: '+e.message);alert('Validation error: '+e.message)}}
async function previewPayload(tab){try{let r=await api('/api/payload/preview/'+tab,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentPayload(tab))});let ta=$(tab+'_payload_preview');if(ta){ta.value=JSON.stringify(r,null,2);ta.closest('details').open=true}renderValidation(tab,r.validation||{});log(`${tab} payload preview updated`)}catch(e){log('Payload preview error: '+e.message);alert('Payload preview error: '+e.message)}}
async function loadPresets(){try{let r=await api('/api/presets');window.PRESETS=r.presets||{};for(const tab of ['inf','wan','h3','samimate']){let sel=$(tab+'_preset_select');if(!sel)continue;let cur=sel.value;let items=Object.keys(window.PRESETS[tab]||{}).sort();sel.innerHTML='<option value="">Preset</option>'+items.map(n=>`<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('');sel.value=cur}}catch(e){log('Preset load error: '+e.message)}}
async function saveCurrentPreset(tab){let name=prompt('Preset name:');if(!name)return;await api('/api/presets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tab,name,settings:gatherSettings()[tabSettingsKey(tab)]})});await loadPresets();set(tab+'_preset_select',name);log('Saved preset: '+name)}
async function applyPreset(tab){let name=val(tab+'_preset_select');if(!name)return alert('Choose a preset first.');let key=tabSettingsKey(tab),next={};next[key]=(window.PRESETS?.[tab]||{})[name]||{};SETTINGS={...SETTINGS,...next};if(tab==='h3')applyH3Settings(SETTINGS.h3||{},SETTINGS.h3_storage||{});else applySettings(false);await persistSettingsQuietly();RESTORED_TABS.delete(tab);await queueTabPreviewRestore(tab,true);log('Loaded preset: '+name)}
async function deletePreset(tab){let name=val(tab+'_preset_select');if(!name)return alert('Choose a preset first.');await api('/api/presets/'+encodeQS(tab)+'/'+encodeQS(name),{method:'DELETE'});await loadPresets();log('Deleted preset: '+name)}
async function loadEndpointProfiles(){try{let r=await api('/api/endpoint-profiles');window.ENDPOINT_PROFILES=r.profiles||{};let sel=$('endpoint_profile_select');if(sel)sel.innerHTML='<option value="">Profile</option>'+Object.keys(window.ENDPOINT_PROFILES).sort().map(n=>`<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('')}catch(e){log('Endpoint profile load error: '+e.message)}}
async function saveEndpointProfile(){let name=prompt('Endpoint profile name:');if(!name)return;let body={name,infinite_endpoint_id:val('infinite_endpoint_id'),wan_endpoint_id:val('wan_endpoint_id'),h3_endpoint_id:val('h3_endpoint_id'),delivery_mode:val('delivery_mode'),s3_bucket:val('s3_bucket'),s3_prefix:val('s3_prefix'),runpod_volume_root:val('runpod_volume_root'),h3_storage:{network_volume_id:val('h3_network_volume_id'),s3_endpoint_url:val('h3_s3_endpoint_url'),s3_bucket:val('h3_s3_bucket'),s3_region:val('h3_s3_region'),s3_prefix:val('h3_s3_prefix'),runpod_volume_root:val('h3_runpod_volume_root')}};await api('/api/endpoint-profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await loadEndpointProfiles();set('endpoint_profile_select',name);log('Saved endpoint profile: '+name)}
async function applyEndpointProfile(){let name=val('endpoint_profile_select'),p=window.ENDPOINT_PROFILES?.[name];if(!p)return alert('Choose a profile first.');for(const k of Object.keys(p)){if(k!=='h3_storage')set(k,p[k])}if(p.h3_storage){let hs=p.h3_storage;set('h3_network_volume_id',hs.network_volume_id);set('h3_s3_endpoint_url',hs.s3_endpoint_url);set('h3_s3_bucket',hs.s3_bucket);set('h3_s3_region',hs.s3_region);set('h3_s3_prefix',hs.s3_prefix);set('h3_runpod_volume_root',hs.runpod_volume_root)}await saveSettings();log('Applied endpoint profile: '+name)}
async function deleteEndpointProfile(){let name=val('endpoint_profile_select');if(!name)return alert('Choose a profile first.');await api('/api/endpoint-profiles/'+encodeQS(name),{method:'DELETE'});await loadEndpointProfiles();log('Deleted endpoint profile: '+name)}
async function loadJobHistory(){try{let r=await api('/api/jobs/history');let el=$('job_history');if(el)el.innerHTML=(r.items||[]).slice(0,30).map(j=>`<div class="jobHistoryItem"><strong>${escapeHtml(j.status||'')}</strong> ${escapeHtml(j.target||'')} ${escapeHtml(j.job_id||'')}<small>${escapeHtml(j.time||'')}</small></div>`).join('')}catch(e){log('Job history load error: '+e.message)}}
let OUTPUT_SYNC_TIMER=null;
async function checkOutputSync(start=false){
  if(OUTPUT_SYNC_TIMER){clearTimeout(OUTPUT_SYNC_TIMER);OUTPUT_SYNC_TIMER=null}
  try{
    const result=await api('/api/outputs/sync',start?{method:'POST'}:{});
    const el=$('output_sync_status');
    if(el)el.textContent=`${result.message||(result.running?'Checking for missing generations…':'Output check complete.')} ${result.downloaded||0} downloaded, ${result.existing||0} already local.${result.errors?.length?' '+result.errors.join(' '):''}`;
    if(result.running)OUTPUT_SYNC_TIMER=setTimeout(()=>checkOutputSync(),2000);
    else await refreshRecentOutputs(false);
  }catch(e){const el=$('output_sync_status');if(el)el.textContent='Output check failed: '+e.message+' — Refresh to retry.'}
}
async function refreshRecentOutputs(announce=true){if(announce)void checkOutputSync(true);try{let r=await api('/api/outputs/recent?limit=12');let el=$('recent_outputs');if(el)showOutputGrid(el,r.items||[]);if(announce)log('Recent outputs refreshed; checking for missing generations')}catch(e){if(announce)log('Recent outputs error: '+e.message)}}
function showOutputGrid(el,items){el.innerHTML='';for(const it of items||[]){let card=document.createElement('div');card.className='outputItem';let url=cacheBustUrl(it.url||('/api/file?path='+encodeQS(it.path)));let media=it.kind==='video'?`<video controls muted preload="none" src="${url}"></video>`:it.kind==='image'?`<img loading="lazy" src="${url}" alt="">`:it.kind==='audio'?`<audio controls preload="none" src="${url}"></audio>`:`<div class="outputFile">File</div>`;card.innerHTML=`${media}<input readonly value="${escapeHtml(it.path||'')}"><div class="trimMeta">${escapeHtml(it.modified||'')} ${it.size_mb?escapeHtml(it.size_mb+' MB'):''}</div>`;el.appendChild(card)}}
async function repairSelectedMedia(action){let path=BROWSER_SELECTED||'';if(!path)return alert('Select a media file first.');try{let r=await api('/api/media/repair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,action,width:+val('repair_width')||0,height:+val('repair_height')||0,fps:+val('repair_fps')||0})});$('repair_status').innerHTML=`<div class="validationItem good">Saved ${escapeHtml(r.path)}</div>`;log('Media repair saved: '+r.path);BROWSER_SELECTED=r.path;BROWSER_SELECTED_ITEM={path:r.path,source:'local',key:''};await selectBrowserFile(null,r.path,'','local');await refreshRecentOutputs(false)}catch(e){$('repair_status').innerHTML=`<div class="validationItem bad">${escapeHtml(e.message)}</div>`;log('Media repair error: '+e.message);alert('Media repair error: '+e.message)}}
function workflowUseCurrentInputs(){if(val('wan_video_path')&&!val('samimate_video_path'))set('samimate_video_path',val('wan_video_path'));if(val('wan_image_path')&&!val('samimate_image_path'))set('samimate_image_path',val('wan_image_path'));previewPayload('samimate');persistSettingsQuietly();log('Workflow synced current inputs into SAMimate')}
async function explainAndLogError(text){try{let r=await api('/api/error/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});(r.hints||[]).forEach(h=>log('Hint: '+h))}catch{}}
const TOOLTIP_HELP={
workflow_status:'Shows validation results for the guided subject-replacement workflow.',
job_history:'Recent RunPod submissions and terminal states recorded by the local app.',
recent_outputs:'Latest media files found under the configured Output Dir.',
inf_tts_text:'Text that will be converted into a speech audio file for InfiniteTalk.',
inf_tts_engine:'Speech generator for this inline InfiniteTalk TTS panel.',
inf_tts_voice:'Windows SAPI voice to use when Windows voices are selected.',
inf_tts_rate:'Speech rate. Negative is slower, positive is faster.',
inf_tts_volume:'Speech volume from 0 to 100.',
tts_text:'Text that will be converted into a standalone speech audio file.',
tts_engine:'Free local TTS engine. Windows voices are built in; Piper uses the configured local model.',
tts_voice:'Voice used by Windows SAPI. Piper voice is selected by the configured model path.',
tts_rate:'Speech rate. Negative is slower, positive is faster.',
tts_volume:'Speech volume from 0 to 100.',
tts_output_path:'Most recent generated TTS audio file.',
inf_tts_output_path:'Most recent inline InfiniteTalk TTS audio file.',
wan_image_path:'Reference image for the identity/appearance WanAnimate should apply.',
wan_video_path:'Driving/reference video that provides motion, pose, timing, and default resolution/FPS.',
wan_mask_path:'Primary black/white edit mask. For subject replacement, use SAM Mask Video here.',
wan_aligned_mask_path:'Local preview path for the mask after applying the current Wan trim, crop, frame cap, output size, and processed video duration.',
wan_aligned_mask_preview:'Video preview of the subject mask exactly as it will be sent to WanAnimate after local alignment and duration conforming.',
wan_aligned_mask_image_preview:'Image preview of the subject mask after resizing to the Wan output size.',
wan_video_end:'Trim end time for the Wan video. The mask video is conformed to this selected input video window before submit.',
wan_masked_video_path:'Optional visual preview video. This is not the preferred Wan edit mask.',
wan_width:'Output width. 0 or blank means use the loaded input video width, rounded for Wan.',
wan_height:'Output height. 0 or blank means use the loaded input video height, rounded for Wan.',
wan_seed:'Deterministic generation seed. Use the dice button for a valid random seed.',
wan_fps:'Output frame rate sent to WanAnimate. Defaults can be loaded from the driving video.',
wan_cfg:'Classifier-free guidance strength sent to WanAnimate.',
wan_steps:'Inference step count sent to WanAnimate.',
wan_prompt:'Prompt describing the desired replacement subject or animation result.',
wan_negative_prompt:'Things to avoid in the generated result.',
wan_advanced:'JSON object merged into the WanAnimate payload for endpoint-specific options.',
samimate_video_path:'Original source video. SAM3 segments this and the final composite uses it as background.',
samimate_image_path:'Primary identity and outfit reference for the replacement person.',
samimate_subject_prompt:'Short SAM3 concept to segment, usually person, woman, man, face, or subject.',
samimate_prompt:'Optional replacement details. H3 automatically builds the complete masked replacement prompt.',
samimate_negative_prompt:'Things to avoid in the SAMimate WanAnimate pass.',
samimate_advanced:'JSON object merged into the SAMimate WanAnimate payload.',
samimate_frame_cap:'Maximum frames used by SAM3 and the SAMimate WanAnimate pass. 0 processes the full selected trim window. Shorter masks are padded/conformed before Wan and composite.',
samimate_delivery_override:'Delivery mode used only by the SAMimate WanAnimate submission.',
samimate_use_wan_points:'Also send current WanAnimate control point payload with SAMimate.',
sam_canvas:'Preview frame editor for SAM points and boxes.',
sam_video_start:'Trim start used by SAM Studio video preview and video mask generation.',
sam_video_end:'Trim end used by SAM Studio video preview and video mask generation.',
sam_payload_preview:'Exact local SAM request JSON before running the mask backend.',
sam_mask_video_path:'Black/white video mask generated by SAM. Use this as WanAnimate mask input.',
sam_masked_video_path:'Visual masked-preview video. Use for checking the mask, not as the main Wan edit mask.',
browser_source:'Choose local filesystem browsing or configured S3 bucket browsing.',
browse_path:'Local folder path or S3 prefix. Blank S3 prefix lists the bucket root.',
browser_sort_by:'Column used to sort the media browser list.',
browser_sort_dir:'Ascending or descending sort order.',
browser_search:'Filename search within the current folder or prefix.',
repair_width:'Target width for Resize repair. Rounded to a video-safe multiple when encoded.',
repair_height:'Target height for Resize repair. Rounded to a video-safe multiple when encoded.',
repair_fps:'Target frame rate for FPS normalization.',
repair_status:'Shows the result of the most recent media repair action.',
endpoint_profile_select:'Saved endpoint/S3 routing profile to apply or delete.',
log:'Live application log. Drag the top edge to resize, or hide it with the Hide button.'
};
function tooltipTextFor(el){if(el.title)return el.title;const id=el.id||'';const txt=(el.textContent||el.placeholder||id).trim();if(TOOLTIP_HELP[id])return TOOLTIP_HELP[id];const map={runpod_api_key:'RunPod API key used to submit jobs to your endpoints.',infinite_endpoint_id:'InfiniteTalk RunPod endpoint ID.',wan_endpoint_id:'WanAnimate RunPod endpoint ID.',timeout_seconds:'Maximum seconds to wait before RunPod kills the job.',delivery_mode:'Global media delivery. Auto matches the old desktop app. RunPod Volume Path is the old path-based delivery wording.',s3_endpoint_url:'RunPod S3-compatible endpoint URL.',s3_bucket:'RunPod network volume bucket name.',s3_prefix:'S3 folder prefix for uploaded input files.',runpod_volume_root:'Path mounted inside the endpoint container, usually /runpod-volume.',output_dir:'Local folder where completed outputs are downloaded. Cache cleanup does not delete this folder.',cache_dir:'Working cache folder for uploads, temp transcodes, and thumbnails.',sam_output_dir:'Folder where SAM frames, masks, overlays, cutouts, masked images, and masked videos are written.',cache_max_age_days:'App cache files older than this many days are deleted when cleanup runs.',cache_cleanup_enabled:'Run app cache cleanup on startup. Use Clean expired now for an immediate cleanup.',hf_cache_dir:'Optional Hugging Face model cache for local SAM3. This is not auto-deleted.',wan_delivery_override:'WanAnimate-only delivery override. Auto matches the old desktop app.',wan_mode:'WanAnimate workflow mode.',wan_pose:'Sends pose_estimation to WanAnimate. This affects the endpoint workflow, not local preprocessing.',wan_face:'Sends face_detection to WanAnimate. This affects the endpoint workflow when supported.',wan_mask_path:'Main subject replacement mask. For video person swaps, use SAM Mask Video here: white subject gets replaced, black background stays.',wan_mask_edit:'Sends mask_editing=true to WanAnimate. Enable when you want the workflow to edit only the supplied subject mask region.',wan_masked_video_path:'Optional visual/composited preview video. Usually not the correct replacement mask; use Mask Video in the mask field instead.',wan_network_volume:'Sends network_volume to WanAnimate so outputs can be written on the endpoint volume.',wan_control_points:'Enable manual point-control payload fields.',wan_points_store:'Raw points_store JSON sent to WanAnimate.',wan_coordinates:'Raw positive coordinate JSON sent to WanAnimate.',wan_neg_coordinates:'Raw negative coordinate JSON sent to WanAnimate.',inf_input_type:'Choose whether InfiniteTalk uses an image or video reference.',inf_person_count:'Single or multi-person talking workflow.',inf_max_frame:'Endpoint-level max frame option.',browser_type:'Filter file browser by media type.',browser_search:'Search within the current browser folder.',browser_recursive:'Include subfolders. This may be slower.',browser_favorites:'Saved favorite folders.',sam_source_path:'Image or video source for SAM Studio.',sam_backend:'Local SAM3 runs Meta SAM3. Shape Test is not SAM3.',sam3_python:'Python executable for the local SAM3 environment.',sam_hf_token:'Optional Hugging Face token for facebook/sam3 gated checkpoint access.',sam_prompt_mode:'Text prompt is usually enough for subject replacement; add points/box when SAM grabs too much or too little.',sam_text_prompt:'For person replacement, use a simple target like person, woman, man, face, or subject. Points/box can refine it.',sam_external_command:'Command template for a real SAM3 runner. The bundled template is only a shape demo.',sam_make_masked_video:'Recommended for subject replacement. Exports black/white Mask Video for Wan plus a separate visual preview video.',sam_video_frame_cap:'Maximum source frames to process for SAM video masks. Set 0 to process the full video length; use a smaller number for quick tests.'};if(map[id])return map[id];if(el.matches?.('button'))return txt?'Runs: '+txt:'';if(el.matches?.('input,textarea'))return txt?'Enter or review '+txt+'.':'';if(el.matches?.('select'))return txt?'Choose '+txt+'.':'';if(el.matches?.('label'))return txt?'Setting: '+txt:'';return txt||''}
function applyTooltips(root=document){if(!$('tooltip_bubble')){const b=document.createElement('div');b.id='tooltip_bubble';b.className='tooltipBubble';document.body.appendChild(b)}const bubble=$('tooltip_bubble');root.querySelectorAll?.('button,input,select,textarea,label,.drop,.timeline,.audioWave,.browserRow,.outputItem,.validationList,canvas,video,audio,[title]').forEach(el=>{const msg=tooltipTextFor(el);if(!msg)return;el.dataset.tip=msg;el.removeAttribute('title');if(el.dataset.tipBound)return;el.dataset.tipBound='1';el.addEventListener('mouseenter',e=>{bubble.textContent=el.dataset.tip;bubble.classList.add('show');positionTip(e)});el.addEventListener('mousemove',positionTip);el.addEventListener('mouseleave',()=>bubble.classList.remove('show'));el.addEventListener('focus',e=>{bubble.textContent=el.dataset.tip;bubble.classList.add('show');positionTip(e)});el.addEventListener('blur',()=>bubble.classList.remove('show'))});function positionTip(e){let x=e.clientX||e.target?.getBoundingClientRect?.().left||20,y=e.clientY||e.target?.getBoundingClientRect?.().bottom||20;bubble.style.left=Math.min(x+14,window.innerWidth-380)+'px';bubble.style.top=Math.min(y+18,window.innerHeight-90)+'px'}}

async function loadTtsVoices(){try{let r=await api('/api/tts/voices'),tts=SETTINGS.tts||{};for(const id of ['tts_voice','inf_tts_voice']){let s=$(id);if(!s)continue;let current=tts.voice||s.value||'';s.innerHTML='<option value="">Default voice</option>'+(r.voices||[]).map(v=>`<option value="${escapeHtml(v.name)}">${escapeHtml(v.name)}${v.culture?' - '+escapeHtml(v.culture):''}</option>`).join('');s.value=current}}catch(e){log('TTS voices error: '+e.message)}}
function ttsBase(prefix){return prefix==='inf'?'inf_tts':prefix}
function ttsPayload(prefix){let b=ttsBase(prefix);return {text:val(b+'_text'),engine:val(b+'_engine')||'windows_sapi',voice:val(b+'_voice'),rate:+(val(b+'_rate')||0),volume:+(val(b+'_volume')||100),piper_exe:val('tts_piper_exe'),piper_model:val('tts_piper_model'),piper_config:val('tts_piper_config')}}
async function loadPiperStatus(){let el=$('piper_status');if(!el)return;try{let r=await api('/api/tts/piper/status');if(!val('tts_piper_model')&&r.model_exists)set('tts_piper_model',r.model_path);if(!val('tts_piper_config')&&r.config_exists)set('tts_piper_config',r.config_path);el.textContent=r.package_available?(r.model_exists?'Ready: '+r.model_path:'Package installed; voice model missing'):'Piper package missing'}catch(e){el.textContent='Piper status failed';log('Piper status error: '+e.message)}}
async function setupPiperVoice(){let el=$('piper_status'),btn=document.activeElement;try{if(btn)btn.disabled=true;if(el)el.textContent='Setting up Piper voice...';let r=await api('/api/tts/piper/setup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});set('tts_default_engine','piper');set('tts_engine','piper');set('inf_tts_engine','piper');set('tts_piper_model',r.model_path);set('tts_piper_config',r.config_path);if(el)el.textContent='Ready: '+r.model_path;await persistSettingsQuietly();log('Piper voice ready: '+r.model_path)}catch(e){if(el)el.textContent='Piper setup failed';log('Piper setup error: '+e.message);alert('Piper setup error: '+e.message)}finally{if(btn)btn.disabled=false}}
async function generateTts(prefix,target){let b=ttsBase(prefix),status=$(b+'_status'),btn=document.activeElement;try{let p=ttsPayload(prefix);if(!p.text.trim())return alert('Enter text to generate speech.');if(btn)btn.disabled=true;if(status)status.textContent='Generating speech...';await saveSettings();let stopProgress=startProgressLogger('tts','Text to Speech generation',estimateSecFor('tts',p),'Generating speech...');let r;try{r=await api('/api/tts/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});stopProgress('speech generated')}catch(e){stopProgress('speech failed');throw e}set(b+'_output_path',r.path);let a=$(b+'_preview');if(a){a.src=cacheBustUrl(r.url);a.classList.remove('hidden');a.load()}rememberPathForInput(target||b+'_output_path',r.path);showOutputPreview('tts',[{path:r.path,url:r.url,kind:'audio',label:`${r.engine} speech${r.duration_seconds?' - '+r.duration_seconds.toFixed(2)+'s':''}`}],true,{title:'Text to Speech'});if(target){set(target,r.path);previewForInput(target,r.url);if(target==='inf_audio2_path')set('inf_person_count','multi');updateInfiniteModeUI();log('Generated TTS and set '+target+': '+r.path)}else{log('Generated TTS: '+r.path)}await persistSettingsQuietly();if(status)status.textContent=`Saved ${r.duration_seconds?r.duration_seconds.toFixed(2)+'s ':''}${r.engine} audio`}catch(e){if(status)status.textContent='TTS error';log('TTS error: '+e.message);alert('TTS error: '+e.message)}finally{if(btn)btn.disabled=false}}
function copyPromptToTts(prefix){set(prefix+'_tts_text',val(prefix+'_prompt'));log('Copied prompt text to TTS script')}
async function copyTtsToInfinite(target){let p=val('tts_output_path');if(!p)return alert('Generate speech first.');set(target,p);previewForInput(target,await previewUrlForPath(p));if(target==='inf_audio2_path')set('inf_person_count','multi');updateInfiniteModeUI();await persistSettingsQuietly();showTab('inf');log('TTS audio sent to '+target)}

// ------------------------- SAM Studio -------------------------
let SAM={img:null,frame:null,box:null,points:{positive:[],negative:[]},dragging:false,dragStart:null,result:null,view:null};
function updateSamBackendNotice(){let backend=val('sam_backend'),isReal=backend==='local_sam3'||backend==='external_sam3',n=$('sam_backend_notice'),b=$('sam_run_button');if(n){n.textContent=backend==='local_sam3'?'Local SAM3 selected. This uses Meta SAM3 through the SAM3 Python path below.':backend==='external_sam3'?'External SAM3 runner selected. Masks depend entirely on the configured command output.':'Shape Test selected. This is not SAM3; it only draws simple preview masks.';n.classList.toggle('real',isReal)}if(b)b.textContent=backend==='local_sam3'?'Run Local SAM3':backend==='external_sam3'?'Run SAM3 Runner':'Run Shape Test'}
function samMode(){return val('sam_prompt_mode')||'points'}function samModeHasBox(){return ['box','text_box','all'].includes(samMode())}function samModeHasPoints(){return ['points','text_points','all'].includes(samMode())}
async function samUseWanImage(){set('sam_source_path',val('wan_image_path'));set('sam_video_start','');set('sam_video_end','');await persistSettingsQuietly();samLoadFrame()}
async function samUseWanVideo(){set('sam_source_path',val('wan_video_path'));set('sam_video_start',val('wan_video_start'));set('sam_video_end',val('wan_video_end'));set('sam_frame_time','0');await persistSettingsQuietly();samLoadFrame()}
async function samUseInfiniteInput(){let isVideo=val('inf_input_type')==='video';set('sam_source_path',val(isVideo?'inf_video_path':'inf_image_path'));set('sam_video_start',isVideo?val('inf_video_start'):'');set('sam_video_end',isVideo?val('inf_video_end'):'');set('sam_frame_time','0');await persistSettingsQuietly();samLoadFrame()}
function samMapPoint(ev){let c=$('sam_canvas'),r=c.getBoundingClientRect(),x=(ev.clientX-r.left)*(c.width/r.width),y=(ev.clientY-r.top)*(c.height/r.height),v=SAM.view;if(!v)return null;if(x<v.x||y<v.y||x>v.x+v.w||y>v.y+v.h)return null;return {x:Math.round(((x-v.x)/v.s)*1000)/1000,y:Math.round(((y-v.y)/v.s)*1000)/1000}}
function samCanvasPoint(pt){let v=SAM.view;return {x:v.x+pt.x*v.s,y:v.y+pt.y*v.s}}
function samPayload(){return {source_path:val('sam_source_path'),source_type:val('sam_source_type'),frame_time:val('sam_frame_time')||0,video_start:val('sam_video_start'),video_end:val('sam_video_end'),backend:val('sam_backend'),sam3_python:val('sam3_python'),hf_token:val('sam_hf_token'),external_command:val('sam_external_command'),prompt_mode:val('sam_prompt_mode'),text_prompt:val('sam_text_prompt'),mask_fill:val('sam_mask_fill'),invert:chk('sam_invert'),make_masked_video:chk('sam_make_masked_video'),video_frame_cap:intFieldOrDefault('sam_video_frame_cap',180),points:SAM.points,box:SAM.box||{}}}
function samUpdatePayloadPreview(){if($('sam_payload_preview'))$('sam_payload_preview').value=JSON.stringify(samPayload(),null,2);if($('sam_status'))$('sam_status').textContent=` +${SAM.points.positive.length} / -${SAM.points.negative.length}${SAM.box?' / box':''}`}
function samDraw(){let c=$('sam_canvas'),ctx=c.getContext('2d');if(SAM.img)fitCanvasToMedia(c,SAM.img.width,SAM.img.height,760,430);ctx.fillStyle='#020617';ctx.fillRect(0,0,c.width,c.height);if(!SAM.img){ctx.fillStyle='#94a3b8';ctx.fillText('Load a source frame to begin',24,34);return}let s=Math.min(c.width/SAM.img.width,c.height/SAM.img.height),w=SAM.img.width*s,h=SAM.img.height*s,x=(c.width-w)/2,y=(c.height-h)/2;SAM.view={x,y,w,h,s,iw:SAM.img.width,ih:SAM.img.height};ctx.drawImage(SAM.img,x,y,w,h);ctx.lineWidth=3;if(SAM.box){let b=SAM.box;ctx.strokeStyle='#38bdf8';ctx.fillStyle='rgba(56,189,248,.16)';let p=samCanvasPoint({x:b.x,y:b.y});ctx.fillRect(p.x,p.y,b.w*s,b.h*s);ctx.strokeRect(p.x,p.y,b.w*s,b.h*s)}let i=1;for(const p of SAM.points.positive){let q=samCanvasPoint(p);ctx.fillStyle='#22c55e';ctx.strokeStyle='white';ctx.beginPath();ctx.arc(q.x,q.y,8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='white';ctx.fillText('+'+i++,q.x+10,q.y-8)}i=1;for(const p of SAM.points.negative){let q=samCanvasPoint(p);ctx.fillStyle='#ef4444';ctx.strokeStyle='white';ctx.beginPath();ctx.arc(q.x,q.y,8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='white';ctx.fillText('-'+i++,q.x+10,q.y-8)}samUpdatePayloadPreview()}
async function samLoadFrame(){try{let p=val('sam_source_path').trim();if(!p)return alert('Choose a SAM source first.');if($('sam_status'))$('sam_status').textContent='Loading source frame...';let r=await api('/api/sam/frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_path:p,frame_time:val('sam_frame_time')||0,video_start:val('sam_video_start'),video_end:val('sam_video_end')})});SAM.frame=r;SAM.img=new Image();SAM.img.onload=()=>{samDraw();log('SAM frame loaded: '+r.frame_path)};SAM.img.onerror=()=>{if($('sam_status'))$('sam_status').textContent='Frame image failed to load';log('SAM frame image failed to load: '+r.frame_url)};SAM.img.src=r.frame_url}catch(e){if($('sam_status'))$('sam_status').textContent='Frame load error';log('SAM frame error: '+e.message)}}
function setupSamStudio(){let c=$('sam_canvas');if(!c)return;c.addEventListener('contextmenu',e=>{e.preventDefault();let p=samMapPoint(e);if(p){SAM.points.negative.push(p);samDraw()}});c.addEventListener('mousedown',e=>{let p=samMapPoint(e);if(!p)return;if(e.button===2||e.shiftKey){SAM.points.negative.push(p);samDraw();return}if(samModeHasBox()&&!e.ctrlKey&&!e.altKey){SAM.dragging=true;SAM.dragStart=p;SAM.box={x:p.x,y:p.y,w:1,h:1};samDraw();return}if(samModeHasPoints()){SAM.points.positive.push(p);samDraw()}});c.addEventListener('mousemove',e=>{if(!SAM.dragging||!SAM.dragStart)return;let p=samMapPoint(e);if(!p)return;let x=Math.min(SAM.dragStart.x,p.x),y=Math.min(SAM.dragStart.y,p.y),w=Math.abs(p.x-SAM.dragStart.x),h=Math.abs(p.y-SAM.dragStart.y);SAM.box={x,y,w,h};samDraw()});window.addEventListener('mouseup',()=>{SAM.dragging=false;SAM.dragStart=null});['sam_prompt_mode','sam_text_prompt','sam_frame_time','sam_video_start','sam_video_end','sam_backend','sam_external_command','sam_mask_fill','sam_invert','sam_make_masked_video'].forEach(id=>$(id)?.addEventListener('change',samUpdatePayloadPreview));['sam_video_start','sam_video_end'].forEach(id=>$(id)?.addEventListener('change',()=>{if(val('sam_source_path').trim())samLoadFrame()}));$('sam_backend')?.addEventListener('change',updateSamBackendNotice);updateSamBackendNotice();samDraw()}
function samClearPoints(){SAM.points={positive:[],negative:[]};samDraw()}function samClearBox(){SAM.box=null;samDraw()}function samClearAll(){SAM.points={positive:[],negative:[]};SAM.box=null;samDraw()}
async function samRun(){let btn=$('sam_run_button');try{let p=samPayload();if(!p.source_path)return alert('Choose a SAM source first.');if(p.backend==='external_sam3'&&!p.external_command.trim())return alert('External SAM3 runner needs a real command. Shape Test is available for UI wiring only.');if(p.backend==='local_sam3'&&!p.text_prompt.trim()&&!SAM.box&&!SAM.points.positive.length)return alert('Local SAM3 needs a text prompt, box, or positive points.');if(btn){btn.disabled=true;btn.textContent='Running...'}$('sam_status').textContent='Running mask backend...';log('Mask backend started: '+p.backend);let stopProgress=startProgressLogger('sam','SAM Studio mask generation',estimateSecFor('sam',p),'Running mask backend...');let r;try{r=await api('/api/sam/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});stopProgress('mask backend complete')}catch(e){stopProgress('mask backend failed');throw e}SAM.result=r;samShowResult(r);let items=samResultItems(r);if(items.length)showOutputPreview('sam',items,true,{title:r.is_real_sam3?'SAM3 Mask Run':'SAM Shape Test',note:(r.backend_used||'local')+(r.mask_count?` masks=${r.mask_count}`:'')});$('sam_status').textContent=(r.is_real_sam3?'SAM3 complete':'Shape test complete');log('Mask backend complete: '+(r.backend_used||'local')+(r.mask_count?` masks=${r.mask_count}`:''));if(r.warning)log('WARNING: '+r.warning)}catch(e){$('sam_status').textContent='Mask error';log('SAM error: '+e.message);alert('SAM error: '+e.message)}finally{if(btn){btn.disabled=false;updateSamBackendNotice()}}}
function samResultItems(r){let items=[];for(const [key,label,kind] of [['mask_video','Mask Video','video'],['inverted_mask_video','Inverted Mask Video','video'],['masked_video','Masked Preview Video','video'],['mask','Mask Image','image'],['overlay','Overlay Image','image'],['cutout','Cutout Image','image'],['masked_image','Masked Image','image']]){let p=r?.[key+'_path'];if(p)items.push({path:p,url:r[key+'_url']||('/api/file?path='+encodeQS(p)),kind,label})}return items}
function samShowResult(r){for(const [id,key] of [['sam_mask','mask'],['sam_overlay','overlay'],['sam_cutout','cutout'],['sam_masked_image','masked_image']]){if(r[key+'_path'])set(id+'_path',r[key+'_path'])}for(const [id,key] of [['sam_mask_video','mask_video'],['sam_masked_video','masked_video']]){let vp=$(id+'_preview');if(r[key+'_path']){set(id+'_path',r[key+'_path']);if(vp){vp.src=cacheBustUrl(r[key+'_url']||('/api/file?path='+encodeQS(r[key+'_path'])));vp.load();vp.classList.remove('hidden')}}else if(vp){vp.removeAttribute('src');vp.classList.add('hidden')}}persistSettingsQuietly()}
async function samSendMaskToWan(){let p=SAM.result?.mask_path||val('sam_mask_path');if(!p)return alert('Run SAM first.');setWanMaskPretrimmed(false);set('wan_mask_path',p);previewForInput('wan_mask_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM mask sent to WanAnimate mask field')}
async function samSendMaskVideoToWan(){let p=SAM.result?.mask_video_path||val('sam_mask_video_path');if(!p)return alert('Enable Make video mask outputs and run SAM on a video first.');setWanMaskPretrimmed(true);set('wan_mask_path',p);previewForInput('wan_mask_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM mask video sent to WanAnimate mask field as an already-trimmed mask')}
async function samSendCutoutToWanImage(){let p=SAM.result?.cutout_path||val('sam_cutout_path');if(!p)return alert('Run SAM first.');set('wan_image_path',p);previewForInput('wan_image_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM cutout sent to Wan image field')}
async function samSendMaskedImageToInf(){let p=SAM.result?.masked_image_path||val('sam_masked_image_path');if(!p)return alert('Run SAM first.');set('inf_input_type','image');updateInfiniteModeUI();set('inf_image_path',p);previewForInput('inf_image_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM masked image sent to InfiniteTalk')}
async function samSendMaskedVideoToInf(){let p=SAM.result?.masked_video_path||val('sam_masked_video_path');if(!p)return alert('Enable Make masked video and run SAM first.');set('inf_input_type','video');updateInfiniteModeUI();set('inf_video_path',p);previewForInput('inf_video_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM masked video sent to InfiniteTalk')}
async function samSendMaskedVideoToWanMask(){let p=SAM.result?.masked_video_path||val('sam_masked_video_path');if(!p)return alert('Enable Make masked video and run SAM first.');set('wan_masked_video_path',p);previewForInput('wan_masked_video_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM masked video sent to Wan masked-video field')}
async function samSendMaskedVideoToWan(){let p=SAM.result?.masked_video_path||val('sam_masked_video_path');if(!p)return alert('Enable Make masked video and run SAM first.');set('wan_video_path',p);previewForInput('wan_video_path',await previewUrlForPath(p));await persistSettingsQuietly();log('SAM masked video sent to Wan reference video field')}

init().catch(e=>log('init error: '+e.message));




function setupH3FrameInputs(){for(const id of ['h3_first_frame_path','h3_last_frame_path']){let path=$(id),card=path?.closest('.h3ReferenceCard');if(!card||card.dataset.frameDropBound)continue;card.dataset.frameDropBound='1';bindH3ReferenceDrop(card,path,'image');path.addEventListener('input',()=>{if(!path.value.trim())clearPreviewForInput(id);persistSettingsSoon()});path.addEventListener('change',async()=>{if(path.value.trim())try{previewForInput(id,await previewUrlForPath(path.value.trim()))}catch(e){log('Frame preview failed: '+e.message)}await persistSettingsQuietly()})}}

function selectH3TurboFamily(family){set('h3_turbo_family',family);set('h3_sampling_preset','custom');if(family==='larry'){fillH3ModelSelect('h3_turbo_lora',[...Array.from($('h3_turbo_lora').options,o=>o.value),'minimax_h3_turbo_v4_step600_ema.safetensors']);set('h3_turbo_lora','minimax_h3_turbo_v4_step600_ema.safetensors');set('h3_sampler','h3_turbo');set('h3_scheduler','simple')}else if(family==='lightx2v'){let ref=val('h3_task')==='r2v',current=val('h3_turbo_lora'),name=current.toLowerCase().includes(ref?'_ref2v_':'_fl2v_')&&current.toLowerCase().includes('_comfyui_')&&!current.toLowerCase().includes('_sla_')?current:ref?'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors':'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors';fillH3ModelSelect('h3_turbo_lora',[...Array.from($('h3_turbo_lora').options,o=>o.value),name]);set('h3_turbo_lora',name);set('h3_sampler','euler');set('h3_scheduler','simple')}updateH3TurboUI(false);persistSettingsSoon()}

// Independent SAMimate controls and reusable tracked masks.
const SAMIMATE_H3_FIELDS=['ref2va_model','text_encoder','video_vae','audio_vae','steps','sampler','scheduler','turbo_enabled','turbo_family','turbo_lora','turbo_strength','cache_enabled','cache_threshold','attention','sparse_keep_percent','sparse_tau','sparse_start_percent','sparse_end_percent','sparse_trained_weights','delivery'];
let SAMIMATE_MASKS=null, SAMIMATE_MASKS_BUSY=false;
function setupSamimateH3Controls(){
  const host=$('samimate_h3_controls');if(!host||host.children.length)return;
  for(const key of SAMIMATE_H3_FIELDS){
    const source=$('h3_'+key),label=source?.closest('label');if(!label)continue;
    const copy=label.cloneNode(true);for(const el of [copy,...copy.querySelectorAll('*')]){for(const a of [...el.attributes])if(a.name.startsWith('on'))el.removeAttribute(a.name);if(el.id)el.id='samimate_'+el.id;el.disabled=false}
    host.appendChild(copy);
  }
  const saved=SETTINGS.samimate||{};
  samimateSetH3Options(saved.h3_options||{...h3Settings(),steps:20,turbo_enabled:false,sampler:'res_multistep',scheduler:'simple',cache_enabled:false,attention:'native'});
  set('samimate_prompt_mode',saved.prompt_mode||'auto');set('samimate_resolution',saved.resolution||'auto');
  for(const key of ['width','height'])$('samimate_'+key).addEventListener('change',()=>set('samimate_resolution','custom'));
  for(const input of document.querySelectorAll('#samimate input[id$="_path"]')){
    const button=document.createElement('button');button.type='button';button.className='pathRecentTrigger';button.textContent='Recent uploads';button.onclick=e=>samimateRecentUploads(input.id,e);input.parentElement.appendChild(button);
    input.addEventListener('focus',()=>showPathRecent(input));
  }
  $('samimate_h3_turbo_family').addEventListener('change',()=>{
    const family=val('samimate_h3_turbo_family');if(family==='larry'){set('samimate_h3_sampler','h3_turbo');samimateSetSelect('turbo_lora','minimax_h3_turbo_v4_step600_ema.safetensors')}
    if(family==='lightx2v'){set('samimate_h3_sampler','euler');if(!val('samimate_h3_turbo_lora').includes('_ref2v_'))samimateSetSelect('turbo_lora','minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors')}
  });
  host.addEventListener('change',persistSettingsSoon);
  for(const id of ['samimate_h3_loras','samimate_prompt_mode','samimate_prompt'])$(id).addEventListener('change',persistSettingsSoon);
  void samimateLoadMaskGallery();
}
function samimateSetSelect(key,value){const el=$('samimate_h3_'+key);if([...el.options].some(o=>o.value===value)){el.value=value;return}fillH3ModelSelect('samimate_h3_'+key,[...Array.from($('samimate_h3_'+key).options,o=>o.value),value]);set('samimate_h3_'+key,value)}
function samimateSetH3Options(options){for(const key of SAMIMATE_H3_FIELDS){const el=$('samimate_h3_'+key);if(!el)continue;const value=options[key];if(value===undefined)continue;if(el.type==='checkbox')el.checked=!!value;else if(el.tagName==='SELECT')samimateSetSelect(key,String(value));else el.value=value}set('samimate_h3_loras',JSON.stringify(options.loras||[],null,2))}
function samimateH3Options(){const result={};for(const key of SAMIMATE_H3_FIELDS){const el=$('samimate_h3_'+key);if(!el)continue;result[key]=el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value}try{result.loras=JSON.parse(val('samimate_h3_loras')||'[]')}catch{result.loras=val('samimate_h3_loras')}return result}
function samimateCopyH3Settings(){samimateSetH3Options(h3Settings());persistSettingsSoon()}
function samimateRefreshModelOptions(){for(const key of ['ref2va_model','text_encoder','video_vae','audio_vae','turbo_lora'])fillH3ModelSelect('samimate_h3_'+key,Array.from($('h3_'+key).options,o=>o.value).filter(v=>!(key==='turbo_lora'&&v.toLowerCase().includes('_fl2v_'))&&!(key==='ref2va_model'&&/_fl2va?_/.test(v.toLowerCase()))))}
async function samimateApplyResolution(){if(val('samimate_resolution')==='auto')await applySamimateSourceDefaults(val('samimate_video_path'));persistSettingsSoon()}
async function samimateEditGeneratedPrompt(){try{const p=samimateH3Payload();if(p.prompt_mode==='custom')p.prompt='';const r=await api('/api/samimate/h3-prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});set('samimate_prompt',r.prompt);set('samimate_prompt_mode','custom');persistSettingsSoon()}catch(e){alert(e.message)}}
function samimateMaskRequest(){const seg=samPayload();return {video_path:val('samimate_video_path'),video_start:val('samimate_video_start'),video_end:val('samimate_video_end'),video_crop:parseCrop('samimate'),video_frame_cap:val('samimate_video_frame_cap'),mask_frame_cap:samimateEffectiveFrameCap(),segmentation:{backend:seg.backend,sam3_python:seg.sam3_python,prompt_mode:'text',text_prompt:val('samimate_subject_prompt')||'person',mask_fill:seg.mask_fill}}}
function samimateMaskRequestKey(){return JSON.stringify(samimateMaskRequest())}
function samimatePreparedPayload(){const p=samimateH3Payload();if(SAMIMATE_MASKS&&SAMIMATE_MASKS.requestKey===samimateMaskRequestKey())return {...p,...SAMIMATE_MASKS.prepared,width:p.width,height:p.height,source_video_path:SAMIMATE_MASKS.prepared.source_path,mask_path:SAMIMATE_MASKS.masks.mask_video_path};return p}
async function samimateShowMasks(result){for(const [id,key] of [['samimate_mask_preview','mask_video_path'],['samimate_inverted_preview','inverted_mask_video_path']]){const path=result.masks[key];if(path){const v=$(id);v.src=await previewUrlForPath(path);v.load()}}$('samimate_mask_status').textContent=result.reused?'Reusing saved masks':'Masks ready'}
async function samimatePrepareMasks(force=false){
  if(SAMIMATE_MASKS_BUSY)return null;SAMIMATE_MASKS_BUSY=true;$('samimate_prepare_button').disabled=true;
  try{await saveSettings();$('samimate_mask_status').textContent='Checking source and preparing masks…';const request=samimateMaskRequest(),requestKey=JSON.stringify(request);const r=await api('/api/samimate/masks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...request,force})});SAMIMATE_MASKS={...r,requestKey};await samimateShowMasks(r);void samimateLoadMaskGallery();return SAMIMATE_MASKS}catch(e){$('samimate_mask_status').textContent='Mask preparation failed';alert(e.message);return null}finally{SAMIMATE_MASKS_BUSY=false;$('samimate_prepare_button').disabled=false}
}
async function samimateLoadMaskGallery(){try{const r=await api('/api/samimate/masks'),host=$('samimate_mask_gallery');if(!host)return;host.replaceChildren();for(const item of r.items){const card=document.createElement('div'),title=document.createElement('p');title.textContent=(item.source.split(/[\\/]/).pop()||'Mask')+' · '+item.created;card.appendChild(title);const video=document.createElement('video');video.controls=true;video.muted=true;video.preload='none';video.className='mediaPreview';video.src=await previewUrlForPath(item.masks.mask_video_path);card.appendChild(video);const button=document.createElement('button');button.textContent=item.legacy?'View mask pair':'Load source, trim and masks';button.onclick=async()=>{if(!item.legacy){set('samimate_video_path',item.source);for(const [k,v] of Object.entries(item.request||{}))set('samimate_'+(k==='mask_frame_cap'?'frame_cap':k),k==='video_crop'?JSON.stringify(v||{}):v??'');set('samimate_subject_prompt',item.segmentation?.text_prompt||'person');if(item.segmentation?.backend)set('sam_backend',item.segmentation.backend);SAMIMATE_MASKS={...item,requestKey:samimateMaskRequestKey()};previewForInput('samimate_video_path',await previewUrlForPath(item.source))}await samimateShowMasks(item)};card.appendChild(button);host.appendChild(card)}if(!r.items.length)host.textContent='No local masks yet.'}catch(e){log('Mask gallery: '+e.message)}}
async function sendSamimateH3(){
  if($('samimate_run_button').disabled||SAMIMATE_STATE?.active||SAMIMATE_MASKS_BUSY)return alert('A SAMimate run or mask preparation is already active.');
  const btn=$('samimate_run_button');btn.disabled=true;
  try{
    if(!samimateReferencePaths().length)throw new Error('Add at least one reference image.');
    await samimateApplyResolution();const masks=await samimatePrepareMasks();if(!masks)return;
    const p=samimatePreparedPayload();if(!p.mask_path)throw new Error('Inputs changed during mask preparation. Prepare masks again.');
    const folder=await api('/api/samimate/run-folder',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const r=await api('/api/run/h3',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});
    SAMIMATE_STATE={active:true,backend:'h3',prepared:masks.prepared,sourceVideo:p.video_path,referenceImage:p.image_path,subjectMask:masks.masks.mask_video_path,invertedMask:masks.masks.inverted_mask_video_path,runDir:folder.path,runName:folder.name,job:r.job.id,endpoint:r.endpoint_id,submittedAt:Date.now()};
    if(r.payload)$('samimate_payload_preview').value=JSON.stringify({endpoint_id:r.endpoint_id,payload:r.payload},null,2);
    samimateSetStatus('H3 submitted',r.job.id);watch(r.endpoint_id,r.job.id,'samimate',{payload:p,estimateSec:estimateSecFor('samimate',p),startedAt:SAMIMATE_STATE.submittedAt});
  }catch(e){samimateSetStatus('Error');alert(e.message)}finally{btn.disabled=false;btn.textContent='Send to H3'}
}

async function samimateRecentUploads(id,event){try{const kind=id==='samimate_video_path'?'video':'image',r=await api('/api/uploads/recent?kind='+kind);const paths=[...new Set([...recentPathsForInput(id),...r.items.map(i=>i.path)])];localStorage.setItem(inputRecentKey(id),JSON.stringify(paths.slice(0,30)));togglePathRecent(id,event)}catch(e){log('Recent uploads: '+e.message)}}

function samimateAutoSize(width,height){let scale=Math.min(1,4096/Math.max(width,height),Math.sqrt(2000000/(width*height)));return {width:Math.max(32,Math.floor(width*scale/32)*32),height:Math.max(32,Math.floor(height*scale/32)*32)}}

const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const source=fs.readFileSync('static/app.js','utf8');
const elements={};const pending=[];
const context={BROWSER_STORAGE:'h3',BROWSER_VISIBLE_ITEMS:[],BROWSER_SELECTED_ITEMS:{},BROWSER_LAST_INDEX:-1,BROWSER_PREVIEW_REQUEST:0,
  $:id=>elements[id]||(elements[id]={classList:{add(){},remove(){}},removeAttribute(){},load(){},src:''}),
  mediaKindFromPath:()=> 'video',updateBrowserSelectionClasses(){},setBrowserSelectedItem(){},encodeQS:encodeURIComponent,
  api:()=>new Promise(resolve=>pending.push(resolve)),cacheBustUrl:x=>x,log(){},browserSelectionSummary:()=>'',navigator:{}};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('async function selectBrowserFile('),source.indexOf('async function setBrowserTarget(')),context);
(async()=>{
 const old=context.selectBrowserFile(null,'old.mp4');const latest=context.selectBrowserFile(null,'latest.mp4');
 pending[1]({kind:'video',file_url:'latest'});await latest;
 pending[0]({kind:'video',file_url:'old'});await old;
 assert.equal(elements.browser_video_preview.src,'latest','late details must not replace selected preview');
 await context.selectBrowserFile({ctrlKey:true},'extra.mp4');
 assert.equal(pending.length,2,'multi-selection must not fetch and load every video');
 console.log('Stale video previews ignored; multi-selection avoids media requests');
})().catch(e=>{console.error(e);process.exitCode=1});

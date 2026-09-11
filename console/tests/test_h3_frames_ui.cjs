const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8');
const uploads = [];
const cards = {};
const paths = {};
for (const id of ['h3_first_frame_path','h3_last_frame_path']) {
  cards[id] = {dataset:{}, classList:{add(){},remove(){}}};
  paths[id] = {id, closest:()=>cards[id], addEventListener(){}};
}
const context = {$:id=>paths[id], uploadFileToTarget:async(file,id,kind)=>uploads.push({file,id,kind}),
  syncH3SubjectDefinitions(){},updateH3ReferenceLabels(){}};
vm.createContext(context);
for (const name of ['bindH3ReferenceDrop','setupH3FrameInputs'])
  vm.runInContext(source.split('\n').find(s=>s.startsWith(`function ${name}(`)),context);
vm.runInContext('setupH3FrameInputs()',context);
(async()=>{
  for (const id of Object.keys(cards)) {
    const file = {name:id+'.png'};
    let prevented = false;
    await cards[id].ondrop({preventDefault(){prevented=true},dataTransfer:{files:[file]}});
    assert.equal(prevented,true);
    assert.equal(uploads.at(-1).file,file);
    assert.equal(uploads.at(-1).id,id);
    assert.equal(uploads.at(-1).kind,'image');
  }
  assert.equal(uploads.length,2);
  console.log('Both H3 frame cards route drops to the correct upload input');
})().catch(e=>{console.error(e);process.exitCode=1});

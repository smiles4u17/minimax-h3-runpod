const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8');
const calls = [];
const elements = {output_sync_status:{textContent:''},recent_outputs:{}};
let running = true;
let scheduled;
const context = {
  $:id=>elements[id],log(){},showOutputGrid(){},clearTimeout(){},
  setTimeout:callback=>{scheduled=callback;return 1},
  api:async(url,opts={})=>{
    calls.push({url,opts});
    return url.includes('/sync')?{running,downloaded:1,existing:14,errors:[]}:{items:[]};
  }
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('let OUTPUT_SYNC_TIMER='),source.indexOf('async function repairSelectedMedia')),context);
(async()=>{
  await context.refreshRecentOutputs();
  assert(calls.some(c=>c.url.endsWith('/sync')&&c.opts.method==='POST'));
  assert(elements.output_sync_status.textContent.includes('1 downloaded, 14 already local'));
  assert(scheduled);
  calls.length=0;
  running=false;
  await scheduled();
  assert(calls.some(c=>c.url.endsWith('/sync')&&!c.opts.method));
  assert(calls.some(c=>c.url.includes('/recent')));
  assert(!calls.some(c=>c.opts.method==='POST'),'poll completion must not start another sync');
  calls.length=0;
  await context.refreshRecentOutputs(false);
  assert(!calls.some(c=>c.url.endsWith('/sync')),'job completion only refreshes the local gallery');
  console.log('Launch/refresh starts catch-up; completion refreshes gallery without restarting sync');
})().catch(e=>{console.error(e);process.exitCode=1});

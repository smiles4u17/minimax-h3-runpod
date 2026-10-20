const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('static/app.js', 'utf8');
const init = source.slice(source.indexOf('async function init(){'), source.indexOf('\n', source.indexOf('async function init(){')));
let ready = false;
const scope = { SETTINGS: {}, Number, localStorage: { getItem: () => null } };
// Run the actual startup sequence with retired controls absent, as in the live page.
for (const match of source.matchAll(/(?:async\s+)?function\s+(\w+)\s*\(/g)) scope[match[1]] = () => {};
scope.$ = id => id.startsWith('inf_') ? null : {addEventListener() {}, textContent: ''};
scope.api = async path => path === '/api/settings' ? {h3: {}, output_dir: ''} : {version: 'test'};
scope.val = () => '';
scope.chk = () => false;
scope.refreshRecentOutputs = () => new Promise(() => {});
scope.log = message => { if (message === 'Web UI ready') ready = true; };
vm.createContext(scope);
vm.runInContext(init + '\ninit()', scope).then(() => {
  assert(ready, 'Startup must reach console-ready so dashboard panels initialize');
  console.log('Console startup completes with retired controls absent');
}).catch(error => { console.error(error); process.exitCode = 1; });

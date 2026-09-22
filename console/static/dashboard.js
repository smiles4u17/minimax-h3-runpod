function initializeDashboard(){
  const key='media-console-card-layout-v1';
  let layouts={};try{layouts=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
  for(const section of document.querySelectorAll('main > .tab')){
    if(section.querySelector(':scope > .dashboardGrid'))continue;
    const tab=section.id,grid=document.createElement('div');grid.className='dashboardGrid';
    const candidates=[];
    for(const child of [...section.children]){
      if(child.matches('.grid.two,.workflowBoard')&&!child.id&&[...child.children].every(x=>x.matches('.card'))){
        for(const c of [...child.children])candidates.push({el:c,width:1});
      }else if(child.matches('.card,details,.h3Section,.outputPanel'))candidates.push({el:child,width:2});
    }
    if(!candidates.length)continue;
    const toolbar=document.createElement('div');toolbar.className='dashboardToolbar';
    toolbar.innerHTML='<span>Click a title bar to expand/collapse. Hold to drag: left/right for half a row; center for full.</span><button type="button">Reset section layout</button>';
    section.insertBefore(toolbar,section.firstChild);section.appendChild(grid);
    let activePointerDrag=null,pendingDrag=null,suppressClick=false;
    // Small grid tracks plus dense placement let later tiles fill vertical holes.
    // Measure natural card heights again after collapse, uploads, previews or resize.
    let packingFrame=0;
    function pack(){
      packingFrame=0;
      if(!grid.getBoundingClientRect().width)return;
      const desktop=window.matchMedia('(min-width:851px)').matches;
      for(const el of grid.children){
        if(el.classList.contains('cardDragging'))continue;
        const rows=desktop?'span '+Math.ceil((el.getBoundingClientRect().height+16)/4):'';
        if(el.style.gridRowEnd!==rows)el.style.gridRowEnd=rows;
      }
    }
    function queuePacking(){if(!packingFrame)packingFrame=requestAnimationFrame(pack)}
    const sizeObserver=new ResizeObserver(queuePacking);
    sizeObserver.observe(grid);
    const contentObserver=new MutationObserver(queuePacking);
    contentObserver.observe(grid,{childList:true});
    window.addEventListener('resize',queuePacking);
    const toolbarHint=toolbar.querySelector('span');
    function clearPending(){if(pendingDrag){clearTimeout(pendingDrag.timer);pendingDrag=null}}
    const records=candidates.map(({el,width},index)=>{
      const title=((el.querySelector('summary strong')||el.querySelector('h2,h3,summary'))?.textContent||'Section '+(index+1)).trim();
      const id=tab+':'+(el.id||title.toLowerCase().replace(/[^a-z0-9]+/g,'-'))+':'+index;
      const saved=(layouts[tab]||[]).find(x=>x.id===id)||{};
      el.dataset.layoutId=id;el.classList.add('dashboardCard');el.dataset.span=String(saved.width||width);el.dataset.side=saved.side||'';
      const bar=document.createElement('div');bar.className='cardLayoutBar';
      const grip=document.createElement('button');grip.type='button';grip.className='cardDragGrip';grip.textContent='⠿ '+title;grip.title='Click to expand/collapse; hold to move';grip.setAttribute('aria-label','Drag section: '+title);grip.draggable=false;
      const actions=document.createElement('div');actions.className='cardLayoutActions';
      for(const [label,titleText,fn] of [
        ['←','Move section earlier',()=>{const prev=el.previousElementSibling;if(prev)grid.insertBefore(el,prev);save()}],
        ['→','Move section later',()=>{const next=el.nextElementSibling;if(next)grid.insertBefore(next,el);save()}],
        ['½','Toggle half or full row',()=>{el.dataset.span=el.dataset.span==='1'?'2':'1';el.dataset.side='';update();save()}],
        ['−','Collapse or expand section',()=>{if(el.tagName==='DETAILS')el.open=!el.open;else el.classList.toggle('dashboardCollapsed');update();save()}]
      ]){const b=document.createElement('button');b.type='button';b.textContent=label;b.title=titleText;b.setAttribute('aria-label',titleText+': '+title);b.onclick=fn;actions.appendChild(b)}
      function update(){actions.children[2].textContent=el.dataset.span==='1'?'½ row':'Full row';actions.children[3].textContent=(el.tagName==='DETAILS'?!el.open:el.classList.contains('dashboardCollapsed'))?'+':'−';actions.children[3].setAttribute('aria-expanded',String(el.tagName==='DETAILS'?el.open:!el.classList.contains('dashboardCollapsed')))}
      bar.append(grip,actions);
      if(el.tagName==='DETAILS'){
        el.querySelector(':scope > summary').appendChild(bar);bar.addEventListener('click',e=>e.preventDefault());
        if(saved.collapsed!==undefined)el.open=!saved.collapsed;
        el.addEventListener('toggle',()=>{update();save()});
      }else{el.prepend(bar);el.classList.toggle('dashboardCollapsed',!!saved.collapsed)}
      // Use pointer dragging rather than native HTML5 DnD: native dragging on a
      // button suppresses pointermove events in several Chromium/browser builds.
      bar.addEventListener('click',e=>{
        if(e.target.closest('.cardLayoutActions'))return;
        e.preventDefault();e.stopPropagation();
        if(suppressClick){suppressClick=false;return}
        if(el.tagName==='DETAILS')el.open=!el.open;else el.classList.toggle('dashboardCollapsed');update();save();
      });
      bar.addEventListener('pointerdown',e=>{
        if(e.target.closest('.cardLayoutActions')||e.button!==0||activePointerDrag)return;
        clearPending();suppressClick=false;e.stopPropagation();
        pendingDrag={pointerId:e.pointerId,timer:setTimeout(()=>{
        pendingDrag=null;
        const rect=el.getBoundingClientRect(),placeholder=document.createElement('div');
        placeholder.className='cardDragPlaceholder';
        placeholder.style.height=Math.max(72,rect.height)+'px';
        placeholder.dataset.span=el.dataset.span;
        grid.insertBefore(placeholder,el);
        activePointerDrag={el,grid,grip,placeholder,pointerId:e.pointerId,offsetX:e.clientX-rect.left,offsetY:e.clientY-rect.top,startX:e.clientX,startY:e.clientY,moved:false,originalNext:el.nextSibling};
        el.classList.add('cardDragging');
        el.style.width=rect.width+'px';el.style.position='fixed';el.style.left=rect.left+'px';el.style.top=rect.top+'px';
        el.style.zIndex='1000';el.style.pointerEvents='none';
        suppressClick=true;bar.setPointerCapture?.(e.pointerId);
        },300)};
      });
      bar.addEventListener('pointerup',finishPointerDrag);
      bar.addEventListener('pointercancel',finishPointerDrag);
      bar.addEventListener('lostpointercapture',finishPointerDrag);
      el.addEventListener('dragover',e=>{if([...e.dataTransfer.types].includes('application/x-console-card')){e.preventDefault();el.classList.add('cardDropTarget')}});
      el.addEventListener('dragleave',e=>{if(!el.contains(e.relatedTarget))el.classList.remove('cardDropTarget')});
      el.addEventListener('drop',e=>{const id=e.dataTransfer.getData('application/x-console-card'),moving=[...grid.children].find(x=>x.dataset.layoutId===id);if(!moving)return;e.preventDefault();e.stopPropagation();if(moving!==el){const after=e.clientY>el.getBoundingClientRect().top+el.offsetHeight/2;grid.insertBefore(moving,after?el.nextSibling:el)}el.classList.remove('cardDropTarget');save()});
      update();return {el,id,index,defaultWidth:width,update};
    });
    const order=(layouts[tab]||[]).map(x=>x.id);
    records.sort((a,b)=>(order.includes(a.id)?order.indexOf(a.id):10000+a.index)-(order.includes(b.id)?order.indexOf(b.id):10000+b.index));
    for(const r of records){grid.appendChild(r.el);sizeObserver.observe(r.el)}
    queuePacking();
    for(const child of [...section.children])if(child.matches('.grid.two,.workflowBoard')&&!child.children.length)child.remove();
    function save(){queuePacking();layouts[tab]=[...grid.children].map(el=>({id:el.dataset.layoutId,width:Number(el.dataset.span),side:el.dataset.side||'',collapsed:el.tagName==='DETAILS'?!el.open:el.classList.contains('dashboardCollapsed')}));localStorage.setItem(key,JSON.stringify(layouts))}
    function finishPointerDrag(e){
      if(pendingDrag&&e.pointerId===pendingDrag.pointerId)clearPending();
      const drag=activePointerDrag;if(!drag||e.pointerId!==drag.pointerId)return;
      if(e.type==='pointercancel'){drag.moved=false;drag.grid.insertBefore(drag.placeholder,drag.originalNext?.parentElement===drag.grid?drag.originalNext:null)}
      if(drag.moved&&drag.dropTarget&&drag.placeholder.dataset.span==='1'){drag.dropTarget.dataset.span='1';drag.dropTarget.dataset.side=drag.placeholder.dataset.side==='left'?'right':'left';}
      if(drag.moved){drag.el.dataset.span=drag.placeholder.dataset.span;drag.el.dataset.side=drag.placeholder.dataset.side||'';}
      drag.grid.insertBefore(drag.el,drag.placeholder);drag.placeholder.remove();
      drag.el.classList.remove('cardDragging');drag.el.style.removeProperty('width');drag.el.style.removeProperty('position');drag.el.style.removeProperty('left');drag.el.style.removeProperty('top');drag.el.style.removeProperty('z-index');drag.el.style.removeProperty('pointer-events');
      queuePacking();
      const moved=drag.moved;activePointerDrag=null;if(moved){suppressClick=true;for(const r of records)r.update();save();}

    }
      toolbar.querySelector('button').onclick=()=>{for(const r of [...records].sort((a,b)=>a.index-b.index)){r.el.dataset.span=String(r.defaultWidth);r.el.dataset.side='';r.el.classList.remove('dashboardCollapsed');if(r.el.tagName==='DETAILS')r.el.open=true;r.update();grid.appendChild(r.el)}save()};
    document.addEventListener('pointermove',e=>{
      const drag=activePointerDrag;if(!drag||e.pointerId!==drag.pointerId)return;
      if(!drag.moved&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<4)return;
      drag.moved=true;e.preventDefault();
      drag.el.style.left=(e.clientX-drag.offsetX)+'px';drag.el.style.top=(e.clientY-drag.offsetY)+'px';
      const target=document.elementFromPoint(e.clientX,e.clientY)?.closest?.('.dashboardCard');
      if(!target||target===drag.el||target.parentElement!==drag.grid)return;
      const rect=target.getBoundingClientRect(),bounds=grid.getBoundingClientRect();
      const fraction=(e.clientX-bounds.left)/bounds.width;
      const zone=fraction<.3?'left':fraction>.7?'right':'center';
      drag.placeholder.dataset.span=zone==='center'?'2':'1';
      drag.placeholder.dataset.side=zone==='center'?'':zone;
      drag.placeholder.textContent=zone==='center'?'Full row':zone==='left'?'Left half':'Right half';
      if(zone==='center'){
        const after=e.clientY>rect.top+rect.height/2;
        grid.insertBefore(drag.placeholder,after?target.nextSibling:target);
      }else{
        drag.dropTarget=target;
        grid.insertBefore(drag.placeholder,zone==='left'?target:target.nextSibling);
      }
      queuePacking();
      if(e.clientY>window.innerHeight-70)window.scrollBy(0,18);
      else if(e.clientY<90)window.scrollBy(0,-18);
    });
    document.addEventListener('pointerup',finishPointerDrag);
    document.addEventListener('pointercancel',finishPointerDrag);
    const cancelDrag=()=>{clearPending();if(activePointerDrag)finishPointerDrag({type:'pointercancel',pointerId:activePointerDrag.pointerId})};
    window.addEventListener('blur',cancelDrag);
    document.addEventListener('keydown',e=>{if(e.key==='Escape')cancelDrag()});
  }
}

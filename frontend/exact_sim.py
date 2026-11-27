"""
exact_sim.py - the approved QubRoute simulation page, driven by real backend data.

    html = render_exact_sim(pg, start, scenarios, mover=..., path=..., window=..., blocked=...,
                            metrics=..., strip=..., theme="dark")
    streamlit.components.v1.html(html, height=EXACT_SIM_HEIGHT, scrolling=True)

Inputs
  pg         PositionGraph-like (needs .graph networkx graph, ._trap_slots dict, .slots_of(trap))
  start      {ion: node} initial placement
  scenarios  [{"name": "Greedy walk", "ops": [...], "costs": [us per op, optional]},
              {"name": "Selective QUBO plan", "ops": [...]}]   (ops via ops_from_greedy / ops_from_plan)
  positions  optional {node: (x, y)}; default is an automatic layout
  metrics    optional dict: kappa, rho, depth, kappa_th, rho_th, depth_th, greedy_cost, lower_bound,
             qubo_vars, aux_vars, solver, solve_ms, triggered
  strip      optional list of 2 dicts {"value": "-9.0%", "label": "..."} shown after the live savings
  theme      "dark" | "light" | "auto"
"""
import json
import re
from typing import Any, Dict, List, Optional, Sequence

EXACT_SIM_HEIGHT = 1380
EXACT_SIM_COMPACT_HEIGHT = 320
_LINEAR_NAME = re.compile(r"^(t\d+:\d+|seg\d+)$")


# ------------------------------------------------------------------ move converters
def ops_from_plan(plan_ops: Sequence[tuple]) -> List[Dict[str, str]]:
    """QUBO MovePlan.ops: ('move', ion, node) / ('swap', ion_a, ion_b)."""
    out = []
    for op in plan_ops:
        if op[0] == "move":
            out.append({"t": "move", "i": str(op[1]), "to": str(op[2])})
        elif op[0] == "swap":
            out.append({"t": "swap", "a": str(op[1]), "b": str(op[2])})
    return out


def ops_from_greedy(pg, start: Dict[Any, Any], actions: Sequence[tuple], mover: Any) -> List[Dict[str, str]]:
    """plan_walk actions: ('move', node) / ('vacate', ion, from, to) / ('swap', ion, slot).
    A greedy swap first walks the mover into the vacated slot, then swaps (so costs match walk_cost)."""
    G = pg.graph
    at = dict(start)
    out: List[Dict[str, str]] = []
    for a in actions:
        kind = a[0]
        if kind == "move":
            out.append({"t": "move", "i": str(mover), "to": str(a[1])})
            at[mover] = a[1]
        elif kind == "vacate":
            _, ion, _frm, to = a
            out.append({"t": "move", "i": str(ion), "to": str(to)})
            at[ion] = to
        elif kind == "swap":
            _, ion, slot = a
            here = at[mover]
            if slot not in G[here]:
                taken = set(at.values())
                step = [n for n in G[here] if n in G[slot] and n not in taken]
                if step:
                    out.append({"t": "move", "i": str(mover), "to": str(step[0])})
                    at[mover] = step[0]
            out.append({"t": "swap", "a": str(mover), "b": str(ion)})
            at[mover], at[ion] = at[ion], at[mover]
    return out


# ------------------------------------------------------------------ layout
def _auto_layout(pg) -> Dict[Any, tuple]:
    G = pg.graph
    traps = getattr(pg, "_trap_slots", {})
    nodes = list(G.nodes())
    if traps and all(_LINEAR_NAME.match(str(n)) for n in nodes):
        pos: Dict[Any, tuple] = {}
        tids = list(traps.keys())
        per_row, gap, row_gap = 4, 3.0, 3.2
        centre = []
        for i in range(len(tids)):
            row, col = divmod(i, per_row)
            if row % 2:
                col = per_row - 1 - col
            centre.append((col * gap, row * row_gap))
        for i, t in enumerate(tids):
            slots = pg.slots_of(t)
            k = len(slots)
            for j, s in enumerate(slots):
                pos[s] = (centre[i][0] + (j - (k - 1) / 2) * 0.9, centre[i][1])
        for i in range(len(tids) - 1):
            name = f"seg{i}"
            if name in G:
                (x1, y1), (x2, y2) = centre[i], centre[i + 1]
                pos[name] = ((x1 + x2) / 2, (y1 + y2) / 2)
        for n in nodes:
            pos.setdefault(n, (0.0, 0.0))
        return pos
    import networkx as nx
    return nx.kamada_kawai_layout(G)


def render_exact_sim(
    pg,
    start: Dict[Any, Any],
    scenarios: List[Dict[str, Any]],
    *,
    mover: Any = None,
    path: Optional[Sequence[Any]] = None,
    window: Optional[Sequence[Any]] = None,
    blocked: Optional[Sequence[Any]] = None,
    positions: Optional[Dict[Any, Sequence[float]]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    strip: Optional[List[Dict[str, str]]] = None,
    theme: str = "auto",
    autoplay: bool = True,
    compact: bool = False,
) -> str:
    G = pg.graph
    pos = positions if positions else _auto_layout(pg)
    node_trap: Dict[Any, str] = {}
    traps = []
    for t in getattr(pg, "_trap_slots", {}).keys():
        slots = list(pg.slots_of(t))
        traps.append({"id": str(t), "slots": [str(s) for s in slots]})
        for s in slots:
            node_trap[s] = str(t)
    data = {
        "nodes": [{"id": str(n), "x": float(pos[n][0]), "y": float(pos[n][1]),
                   "trap": node_trap.get(n)} for n in G.nodes() if n in pos],
        "edges": [[str(u), str(v)] for u, v in G.edges()],
        "traps": traps,
        "start": {str(k): str(v) for k, v in start.items()},
        "scenarios": scenarios,
        "mover": None if mover is None else str(mover),
        "path": [str(p) for p in (path or [])],
        "window": [str(w) for w in (window or [])],
        "blocked": [str(b) for b in (blocked or [])],
        "metrics": metrics,
        "strip": strip or [],
        "theme": theme,
        "autoplay": bool(autoplay),
        "compact": bool(compact),
    }
    return _TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))


_TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap">
<style>
:root{--bg:#040914;--panel:#0b1a38;--ink:#fff;--mute:#bccbe6;--track:#93b4ea;--line:rgba(165,195,240,.6);--gold:#ffd762;--blue:#d3e0f7;--red:#ff6a70;--ok:#6fe9b5;--grid:rgba(255,255,255,.045)}
:root[data-theme=light]{--bg:#eceae3;--panel:#f8f7f3;--ink:#14213d;--mute:#5a6580;--track:#7d8aa3;--line:rgba(20,33,61,.3);--gold:#7f6220;--blue:#9fb4d6;--red:#c23a43;--ok:#0b7f5a;--grid:rgba(20,33,61,.05)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Plus Jakarta Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
 background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);background-size:28px 28px}
.wrap{max-width:1240px;margin:0 auto;padding:14px 16px 24px}
.ctl{display:flex;gap:8px;align-items:center;justify-content:flex-end;flex-wrap:wrap}
button{font:700 14px "Plus Jakarta Sans",system-ui,sans-serif;color:var(--ink);background:transparent;border:1px solid var(--ink);border-radius:5px;padding:8px 15px;cursor:pointer}
button:hover{background:var(--gold);color:#040914;border-color:var(--gold)}
button:focus-visible{outline:3px solid var(--gold);outline-offset:2px}
label{font-size:13px;font-weight:600;color:var(--mute);display:flex;align-items:center;gap:8px}
input[type=range]{accent-color:var(--gold);width:110px}
.score{display:grid;grid-template-columns:repeat(4,1fr);margin-top:14px;border-top:1px solid var(--line);border-bottom:2px solid var(--track)}
.score div{padding:14px 18px;border-left:1px solid var(--line)}.score div:first-child{border-left:0;padding-left:0}
.score b{display:block;font-size:clamp(30px,4.2vw,54px);font-weight:700;letter-spacing:-.03em;color:var(--gold);line-height:1;margin-bottom:6px}
.score span{font-size:13px;color:var(--mute);font-weight:600}
.maps{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-top:18px}
.card{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,0) 55%),var(--panel);border:1px solid var(--line);border-radius:3px;padding:16px 18px}
.card.q{border-color:var(--gold);box-shadow:0 0 0 1px rgba(255,215,98,.3),0 24px 70px -24px rgba(255,215,98,.45)}
:root[data-theme=light] .card.q{box-shadow:0 18px 50px -26px rgba(127,98,32,.5)}
.ch{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.ch h2{margin:0;font-size:18px;font-weight:700}
.ch span{font-size:14px;color:var(--mute);font-weight:600}.ch span i{font-style:normal;color:var(--ink);font-size:22px;font-weight:800;margin:0 3px 0 8px}
svg{display:block;width:100%;height:auto}
.res{margin-top:14px;padding:12px 16px;border:1px solid var(--ok);color:var(--ok);font-weight:700;font-size:15px;display:none;background:var(--panel)}
.res.on{display:block}
.deck{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;margin-top:16px}
.heat{display:grid;margin-top:12px;gap:3px;font-size:12px}
.heat .l{color:var(--mute);font-weight:600;align-self:center;white-space:nowrap}
.hc{height:27px;border:1px solid var(--line);border-radius:3px;display:flex;gap:3px;align-items:center;justify-content:center;opacity:.25;transition:opacity .5s,background .5s}
.hc.on{opacity:1;background:rgba(255,215,98,.1)}
.hc i{width:12px;height:12px;border-radius:50%;box-shadow:0 0 9px currentColor}
.log{height:226px;overflow:auto;font-size:13.5px;line-height:1.5;margin-top:10px}
.log p{margin:0 0 4px;border-left:3px solid;padding-left:9px}
.stats{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin-top:16px}
.g{margin:10px 0}.g .t{display:flex;justify-content:space-between;font-weight:700;font-size:15px}
.bar{height:5px;background:rgba(128,150,190,.25);margin-top:6px;position:relative}.bar u{position:absolute;inset:0 auto 0 0;background:var(--red)}
.fact{margin:8px 0;font-size:14.5px;line-height:1.5}
.legend{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;font-size:14px;font-weight:600;margin-top:10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-1px;margin-right:7px}
.flow{stroke-dasharray:2 12;animation:fl 1.4s linear infinite}@keyframes fl{to{stroke-dashoffset:-28}}
.halo{transform-box:fill-box;transform-origin:center;animation:ha 1.8s ease-in-out infinite}
@keyframes ha{0%,100%{transform:scale(1);opacity:.5}50%{transform:scale(2.1);opacity:0}}
.rip{transform-box:fill-box;transform-origin:center;animation:rp 1s ease-out forwards}
@keyframes rp{from{transform:scale(1);opacity:1}to{transform:scale(3.6);opacity:0}}
.blk{transform-box:fill-box;transform-origin:center;animation:pu 1s ease-in-out infinite;filter:drop-shadow(0 0 7px var(--red))}
@keyframes pu{50%{transform:scale(1.3);opacity:.4}}
.win{animation:ants 1.4s linear infinite}@keyframes ants{to{stroke-dashoffset:-22}}
@media (prefers-reduced-motion:reduce){.flow,.halo,.rip,.blk,.win{animation:none}}
@media (max-width:820px){.score{grid-template-columns:1fr 1fr}.deck,.stats{grid-template-columns:1fr}.maps{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="ctl"><button id="zoombtn" style="display:none">Full chip</button><button id="play">Play</button><button id="step">Step</button><button id="reset">Reset</button>
<label>Speed <input id="spd" type="range" min="0.5" max="3" step="0.25" value="1"></label><button id="theme">Switch theme</button></div>
<div class="score" id="score"></div>
<div class="maps" id="maps"></div><div class="res" id="res"></div>
<div class="deck"><div class="card" style="overflow-x:auto"><div class="ch"><h2 id="hh">Time-expanded QUBO: x(i, v, t)</h2><span id="hs"></span></div><div class="heat" id="heat"></div></div>
<div class="card"><div class="ch"><h2>Operation log</h2><span>newest first</span></div><div class="log" id="log"></div></div></div>
<div class="stats" id="stats"></div>
</div>
<script>
const D=__DATA__;
const NS='http://www.w3.org/2000/svg',COL=['#ffc83d','#6db3ff','#bb9cff','#4fe0b0','#ff8aa4','#55d3f0','#ff9d5c','#9be564','#e6b3ff','#7df0d0'];
const el=(n,a,p)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e};
const $=id=>document.getElementById(id),sleep=ms=>new Promise(r=>setTimeout(r,ms));
const root=document.documentElement;
root.setAttribute('data-theme',D.theme==='dark'?'dark':D.theme==='light'?'light':(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'));
$('theme').onclick=()=>root.setAttribute('data-theme',root.getAttribute('data-theme')==='light'?'dark':'light');
const ionIds=Object.keys(D.start),colOf=i=>COL[ionIds.indexOf(i)%COL.length];
const nodeById={};D.nodes.forEach(n=>nodeById[n.id]=n);
const W=900,PAD=64,HMAX=D.compact?260:(D.nodes.length>40?560:440);
if(D.compact){document.querySelector('.ctl').style.display='none';document.querySelector('.deck').style.display='none';$('stats').style.display='none';$('score').style.display='none';document.querySelector('.wrap').style.padding='4px';$('maps').style.marginTop='0';}
const canZoom=!D.compact&&D.traps.length>12&&D.window.length>0;
let zoomed=canZoom;
if(canZoom){$('zoombtn').style.display='';}
function computeP(){
 const subset=zoomed?D.nodes.filter(n=>D.window.includes(n.id)||D.path.includes(n.id)):D.nodes;
 const src=subset.length?subset:D.nodes;
 const xs=src.map(n=>n.x),ys=src.map(n=>n.y);
 const padCoord=zoomed?140:0;
 const minx=Math.min(...xs)-padCoord,maxx=Math.max(...xs)+padCoord,miny=Math.min(...ys)-padCoord,maxy=Math.max(...ys)+padCoord;
 const bw=(maxx-minx)||1,bh=(maxy-miny)||1;let sc=(W-2*PAD)/bw;if(bh*sc+2*PAD>HMAX)sc=Math.min(sc,(HMAX-2*PAD)/bh);
 const H=Math.min(HMAX,Math.max(190,bh*sc+2*PAD));
 const P={};D.nodes.forEach(n=>P[n.id]=[(n.x-minx)*sc+(W-bw*sc)/2,(n.y-miny)*sc+(H-bh*sc)/2]);
 return {P,H,sc};
}
let {P,H,sc}=computeP();
const isTr=id=>nodeById[id]&&nodeById[id].trap;
const estCost=(o,from)=>o.t==='swap'?300:((!isTr(from)||!isTr(o.to))?90:10);
const panels=[];

/* result strip */
const strip=$('score');const two=D.scenarios.length>1;
strip.innerHTML=(two?'<div><b id="s1">0%</b><span>fewer operations in this event</span></div><div><b id="s2">0</b><span>µs of shuttling time saved (estimated)</span></div>':'')
 +D.strip.map(s=>'<div><b>'+s.value+'</b><span>'+s.label+'</span></div>').join('');
if(!strip.children.length)strip.style.display='none';
strip.style.gridTemplateColumns='repeat('+Math.max(1,strip.children.length)+',1fr)';

function build(sc_,idx){
 const card=document.createElement('div');card.className='card'+(idx>0?' q':'');
 card.innerHTML='<div class="ch"><h2>'+sc_.name+'</h2><span>ops<i class="n">0</i>µs<i class="t">0</i></span></div><div class="map"></div>';
 $('maps').appendChild(card);
 const svg=el('svg',{viewBox:'0 0 '+W+' '+H,role:'img','aria-label':sc_.name});card.querySelector('.map').appendChild(svg);
 const df=el('defs',{},svg),f=el('filter',{id:'g'+idx,filterUnits:'userSpaceOnUse',x:0,y:0,width:W,height:H},df);
 el('feGaussianBlur',{stdDeviation:3.4,result:'b'},f);const mg=el('feMerge',{},f);['b','b','SourceGraphic'].forEach(k=>el('feMergeNode',{in:k},mg));
 const accent=idx>0?'var(--gold)':'var(--blue)';
 const wp=D.window.filter(k=>P[k]).map(k=>P[k]);
 if(wp.length){const x0=Math.min(...wp.map(p=>p[0]))-34,x1=Math.max(...wp.map(p=>p[0]))+34,y0=Math.min(...wp.map(p=>p[1]))-34,y1=Math.max(...wp.map(p=>p[1]))+34;
  el('rect',{class:'win',x:x0,y:y0,width:x1-x0,height:y1-y0,rx:8,fill:'none',stroke:accent,'stroke-width':1.4,'stroke-dasharray':'5 6',opacity:.95},svg);
  const t=el('text',{x:x0+6,y:y0-8,'font-size':12,'font-weight':700,fill:accent},svg);t.textContent='Window W'}
 D.edges.forEach(([a,b])=>{if(P[a]&&P[b])el('line',{x1:P[a][0],y1:P[a][1],x2:P[b][0],y2:P[b][1],stroke:'var(--track)','stroke-width':1.6,'stroke-linecap':'round'},svg)});
 D.edges.forEach(([a,b])=>{if(P[a]&&P[b]&&!(isTr(a)&&isTr(b)))el('line',{class:'flow',x1:P[a][0],y1:P[a][1],x2:P[b][0],y2:P[b][1],stroke:accent,'stroke-width':2.2,'stroke-linecap':'round',opacity:.8,filter:'url(#g'+idx+')'},svg)});
 D.traps.forEach(t=>{const a=P[t.slots[0]],b=P[t.slots[t.slots.length-1]];if(!a||!b)return;
  el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'var(--track)','stroke-width':30,'stroke-linecap':'round',opacity:.9},svg);
  el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'var(--panel)','stroke-width':26,'stroke-linecap':'round'},svg);
  if(D.traps.length<=16||zoomed){const tx=el('text',{x:(a[0]+b[0])/2,y:(a[1]+b[1])/2-24,'text-anchor':'middle','font-size':11,'font-weight':700,fill:'var(--mute)'},svg);tx.textContent=t.id}});
 const showNodeLbl=D.nodes.length<=28||zoomed;
 D.nodes.forEach(n=>{const p=P[n.id];if(!p||p[0]<-40||p[0]>W+40||p[1]<-40||p[1]>H+40)return;
  if(n.trap){const c=el('circle',{cx:p[0],cy:p[1],r:3.6,fill:'var(--panel)',stroke:'var(--track)','stroke-width':1.3},svg);el('title',{},c).textContent=n.id}
  else{const c=el('circle',{cx:p[0],cy:p[1],r:11,fill:'var(--panel)',stroke:'var(--gold)','stroke-width':1.8},svg);el('title',{},c).textContent=n.id;
   if(showNodeLbl){const t=el('text',{x:p[0],y:p[1]+3,'text-anchor':'middle','font-size':8,'font-weight':700,fill:'var(--ink)'},svg);t.textContent=n.id.length>4?'':n.id}}});
 if(D.path.length>1)el('polyline',{points:D.path.filter(k=>P[k]).map(k=>P[k].join(',')).join(' '),fill:'none',stroke:accent,'stroke-width':2,'stroke-dasharray':'3 8',opacity:.5,'stroke-linecap':'round'},svg);
 const gT=el('g',{filter:'url(#g'+idx+')'},svg),gB=el('g',{},svg),gI=el('g',{},svg);
 const blk=D.blocked.filter(k=>P[k]).map(k=>el('circle',{class:'blk',cx:P[k][0],cy:P[k][1],r:18,fill:'none',stroke:'var(--red)','stroke-width':2},gB));
 const showIonLbl=!zoomed&&D.nodes.length>45?false:true;
 const ions={};ionIds.forEach(i=>{const g=el('g',{},gI),c=colOf(i);
  el('circle',{class:'halo',r:11,fill:c,opacity:.45},g);
  el('circle',{r:11,fill:c,stroke:'rgba(255,255,255,.9)','stroke-width':1.4,style:'filter:drop-shadow(0 0 4px '+c+') drop-shadow(0 0 12px '+c+')'},g);
  if(showIonLbl){const t=el('text',{'text-anchor':'middle',y:3.2,'font-size':9,'font-weight':800,fill:'#0a1730'},g);t.textContent='Q'+i;}
  el('title',{},g).textContent='Q'+i;
  if(i===D.mover)el('circle',{r:16,fill:'none',stroke:accent,'stroke-width':1.4},g);ions[i]=g});
 const S={ops:sc_.ops,costs:sc_.costs||[],name:sc_.name,at:{...D.start},i:0,t:0,card,gT,blk,ions,
  place(){ionIds.forEach(i=>ions[i].setAttribute('transform','translate('+P[S.at[i]]+')'))},
  reset(){S.at={...D.start};S.i=0;S.t=0;gT.innerHTML='';blk.forEach(b=>b.style.display='');S.place();card.querySelector('.n').textContent=0;card.querySelector('.t').textContent=0},
  run(dur){return new Promise(r=>{const o=S.ops[S.i];blk.forEach(b=>b.style.display='none');
   const mv=o.t==='move'?[[o.i,P[o.to],0]]:[[o.a,P[S.at[o.b]],1],[o.b,P[S.at[o.a]],-1]],from=mv.map(m=>P[S.at[m[0]]]),fromNode=S.at[o.t==='move'?o.i:o.a],t0=performance.now();
   (function f(n){const u=Math.min(1,(n-t0)/dur),e=u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
    mv.forEach((m,k)=>ions[m[0]].setAttribute('transform','translate('+(from[k][0]+(m[1][0]-from[k][0])*e)+','+(from[k][1]+(m[1][1]-from[k][1])*e+m[2]*Math.sin(Math.PI*u)*18)+')'));
    if(u<1)return requestAnimationFrame(f);
    mv.forEach((m,k)=>{const c=colOf(m[0]);el('line',{x1:from[k][0],y1:from[k][1],x2:m[1][0],y2:m[1][1],stroke:c,'stroke-width':m[0]===D.mover?4:2.2,'stroke-linecap':'round','stroke-dasharray':o.t==='swap'?'1 5':'none',opacity:.9},gT);
     const rp=el('circle',{class:'rip',cx:m[1][0],cy:m[1][1],r:11,fill:'none',stroke:c,'stroke-width':2.5},gT);setTimeout(()=>rp.remove(),1000)});
    if(o.t==='move')S.at[o.i]=o.to;else{const a=S.at[o.a];S.at[o.a]=S.at[o.b];S.at[o.b]=a}
    const c=(S.costs[S.i]!=null)?S.costs[S.i]:estCost(o,fromNode);S.lastCost=c;S.t+=c;S.i++;
    card.querySelector('.n').textContent=S.i;card.querySelector('.t').textContent=Math.round(S.t);r()})(t0)})}};
 S.place();panels.push(S);return S}
D.scenarios.forEach(build);

/* time-expanded grid: only visited rows (max 12) and plan-length columns */
const last=panels[panels.length-1];
const states=[{...D.start}];last.ops.forEach(o=>{const s={...states[states.length-1]};if(o.t==='move')s[o.i]=o.to;else{const a=s[o.a];s[o.a]=s[o.b];s[o.b]=a}states.push(s)});
const movedIons=new Set();last.ops.forEach(o=>{if(o.t==='move')movedIons.add(o.i);else{movedIons.add(o.a);movedIons.add(o.b);}});
let rows=[];
states.forEach(s=>movedIons.forEach(q=>{const k=s[q];if(k&&!rows.includes(k))rows.push(k)}));
if(!rows.length){D.path.forEach(k=>{if(!rows.includes(k))rows.push(k)});states.forEach(s=>Object.values(s).forEach(k=>{if(!rows.includes(k)&&rows.length<12)rows.push(k)}));}
rows=rows.slice(0,12);
const T=states.length;
$('heat').style.gridTemplateColumns='72px repeat('+T+',minmax(26px,1fr))';
let h='<div></div>';for(let t=0;t<T;t++)h+='<div class="l" style="text-align:center">t='+t+'</div>';
rows.forEach(n=>{h+='<div class="l">'+n+'</div>';for(let t=0;t<T;t++){const d=ionIds.filter(q=>states[t][q]===n).map(q=>'<i style="background:'+colOf(q)+';color:'+colOf(q)+'"></i>').join('');h+='<div class="hc" data-t="'+t+'">'+d+'</div>'}});
$('heat').innerHTML=h;
$('hs').textContent=(D.metrics&&D.metrics.qubo_vars!=null?D.metrics.qubo_vars+' decision variables':'trajectory of the '+last.name.toLowerCase());
if(!two)$('hh').textContent='Time-expanded trajectory: x(i, v, t)';

/* stats cards */
const M=D.metrics;let sh='';
if(M){const f=(v,d)=>v==null?'-':Number(v).toFixed(d);
 sh+='<div class="card"><div class="ch"><h2>Congestion event on the path of Q'+(D.mover==null?'?':D.mover)+'</h2><span>window W: '+D.window.length+' positions</span></div>'
 +'<div class="g"><div class="t"><span>Local contention κ</span><span>'+f(M.kappa,3)+(M.kappa_th!=null?' (threshold '+f(M.kappa_th,2)+')':'')+'</span></div><div class="bar"><u style="width:'+Math.min(100,(M.kappa||0)*100)+'%"></u></div></div>'
 +'<div class="g"><div class="t"><span>Routing regret ρ</span><span>'+f(M.rho,3)+(M.rho_th!=null?' (threshold '+f(M.rho_th,2)+')':'')+'</span></div><div class="bar"><u style="width:'+Math.min(100,(M.rho||0)*100)+'%"></u></div></div>'
 +'<div class="g"><div class="t"><span>Blockage depth d</span><span>'+(M.depth==null?'-':M.depth)+(M.depth_th!=null?' (threshold '+M.depth_th+')':'')+'</span></div><div class="bar"><u style="width:'+Math.min(100,(M.depth||0)*33)+'%"></u></div></div>'
 +'<p class="fact">'+(M.triggered===false?'Trigger <b>not fired</b>: the greedy walk is used.':'Trigger <b>fired</b>: greedy needs '+(M.greedy_cost==null?'?':M.greedy_cost)+' ops'+(M.lower_bound!=null?', the lower bound is '+M.lower_bound:'')+'. The QUBO has '+(M.qubo_vars==null?'?':M.qubo_vars)+' decision variables'+(M.aux_vars!=null?' (+'+M.aux_vars+' auxiliary)':'')+(M.solver?', solved by '+M.solver+(M.solve_ms!=null?' in '+f(M.solve_ms,1)+' ms':''):'')+'; every sample is checked against the physical rules.')+'</p></div>'}
sh+='<div class="card"><div class="ch"><h2>Ions</h2><span>one colour per qubit</span></div><div class="legend">'+ionIds.map(q=>'<div><i style="background:'+colOf(q)+'"></i>Q'+q+(q===D.mover?' (mover)':'')+'</div>').join('')+'</div>'
 +'<p class="fact">Blocked positions pulse red. Dashed box: window W. Est. µs: ion swap 300, junction or segment hop 90, in-trap hop 10.</p></div>';
$('stats').innerHTML=sh;if(!M)$('stats').style.gridTemplateColumns='1fr';

/* controller */
let i=0,busy=false,playing=false,logged=0,fin=false;const maxLen=Math.max(0,...panels.map(p=>p.ops.length));
const dsc=o=>o.t==='swap'?'Q'+o.a+' &#8644; Q'+o.b:'Q'+o.i+' &rarr; '+o.to;
function cu(id,to,fm){const e=$(id);if(!e)return;const t0=performance.now();(function g(n){const u=Math.min(1,(n-t0)/1100);e.textContent=fm(Math.round(to*(1-Math.pow(1-u,3))));if(u<1)requestAnimationFrame(g)})(t0)}
function upd(){document.querySelectorAll('.hc').forEach(c=>c.classList.toggle('on',+c.dataset.t<=i));
 if(i===0){$('log').innerHTML='';logged=0;fin=false;if($('s1')){$('s1').textContent='0%';$('s2').textContent='0'}$('res').classList.remove('on')}
 for(;logged<i;logged++){const k=logged;panels.forEach((p,pi)=>{if(k<p.ops.length){const c=pi>0?'var(--gold)':'var(--blue)';
  $('log').insertAdjacentHTML('afterbegin','<p style="border-color:'+c+'"><b style="color:'+c+'">'+(pi>0?'QUBO':'Greedy')+'</b> &nbsp;op '+(k+1)+' &middot; '+dsc(p.ops[k])+'</p>')}})}
 if(i>=maxLen&&!fin){fin=true;result()}}
function result(){if(panels.length<2)return;const a=panels[0],b=panels[1],so=a.i-b.i,st=Math.round(a.t-b.t),r=$('res');r.classList.add('on');
 if(so>0){cu('s1',Math.round(100*so/a.i),v=>'−'+v+'%');cu('s2',st,v=>v);r.textContent=b.name+' saved '+so+' operations and '+st+' est. µs ('+Math.round(100*so/a.i)+'% fewer).'}
 else r.textContent='Both plans cost '+a.i+' operations: the greedy walk is kept.'}
async function stepOnce(){if(busy||i>=maxLen)return;busy=true;const d=900/ +$('spd').value;
 await Promise.all(panels.map(p=>i<p.ops.length?p.run(d):null));i++;busy=false;upd();if(i>=maxLen){playing=false;$('play').textContent='Replay'}}
async function loop(){while(playing&&i<maxLen){await stepOnce();await sleep(220)}}
async function rst(){playing=false;while(busy)await sleep(40);panels.forEach(p=>p.reset());i=0;upd();$('play').textContent='Play'}
$('play').onclick=async()=>{if(!maxLen)return;if(i>=maxLen)await rst();playing=!playing;$('play').textContent=playing?'Pause':'Play';if(playing)loop()};
$('step').onclick=()=>{playing=false;$('play').textContent='Play';stepOnce()};$('reset').onclick=rst;
if(!maxLen){['play','step','reset'].forEach(k=>$(k).style.display='none')}
if(canZoom){$('zoombtn').onclick=async()=>{await rst();zoomed=!zoomed;$('zoombtn').textContent=zoomed?'Full chip':'Zoom window';({P,H,sc}=computeP());$('maps').innerHTML='';panels.length=0;D.scenarios.forEach(build);i=0;upd();};}
upd();
if(D.autoplay&&maxLen)setTimeout(()=>{if(!playing&&i===0){playing=true;$('play').textContent='Pause';loop()}},1200);
</script></body></html>"""

"""SynthWeb: procedurally generated dynamic web pages with exact, live ground truth.

Training data for the System-1 must be labelled without humans and must not come from the evaluation
benchmarks (DynaCU-Bench, OSWorld). Each page is generated from a seed: a random theme (palette, font,
density, radius), one of several layouts (dashboard, form, shop list, settings, article) and a set of
timed or action-triggered dynamics drawn from the event families the question bank asks about:

  dialog      modal / cookie / permission / update dialogs over a dimmed page, dismissed by a button
  toast       short-lived success / info / error toasts (0.6-3 s)
  error       persistent error banners after a failed action
  loading     spinners, skeleton placeholders and progress bars before content appears (0.2-4 s)
  navigation  page switches with a slide / fade transition
  shift       late content (an ad, a banner) pushes the layout down: targets move
  disabled    buttons that become enabled after a delay
  long_op     long progress bars (5-15 s) with a percentage
  ambient     carousels, tickers, animated charts, blinking carets, a clock (benign change)

`window.__gt()` returns the exact current state each frame: page id, a version counter bumped by every
task-relevant change, visible dialog / toast / error boxes, loading, animating (task-relevant transitions,
not ambient motion), long_op, ambient boxes and every interactive element with its box and enabled flag.
The family list, timings and styles are all seeded; splits are made by seed (and by held-out families).
"""

from __future__ import annotations

import json
import random

FAMILIES = ["dialog", "toast", "error", "loading", "navigation", "shift", "disabled", "long_op", "ambient"]
LAYOUTS = ["dashboard", "form", "shop", "settings", "article"]
WORDS = ("account alpha analytics archive backup billing budget calendar campaign cart catalog channel checkout "
         "client cloud cluster comment contract dashboard data delivery deploy device draft export feed file "
         "filter folder forecast gallery group invoice issue job keyword label launch ledger library license "
         "list log market media member message metric module network note order orders page panel partner "
         "payment pipeline plan policy portal post price product profile project queue quota receipt record "
         "region release report request review role sales schedule search server session settings share "
         "shipment signal source stock storage subscription summary support survey sync task team ticket "
         "timeline token traffic transfer trend upload usage user vendor version video wallet webhook widget "
         "workflow workspace").split()
FONTS = ["Arial", "Helvetica", "Verdana", "Georgia", "Trebuchet MS", "Tahoma", "Courier New", "DejaVu Sans"]


def _palette(r: random.Random) -> dict:
    hue = r.randint(0, 359)
    dark = r.random() < 0.25
    return {"bg": f"hsl({hue},{r.randint(5, 30)}%,{8 if dark else r.randint(94, 99)}%)",
            "fg": "#eee" if dark else "#222", "card": f"hsl({hue},{r.randint(5, 25)}%,{16 if dark else 100}%)",
            "accent": f"hsl({(hue + r.choice([0, 30, 180, 210])) % 360},{r.randint(55, 85)}%,{r.randint(40, 55)}%)",
            "muted": "#888", "dark": dark}


def _words(r: random.Random, n: int) -> str:
    return " ".join(r.choice(WORDS) for _ in range(n)).capitalize()


def generate(seed: int, families: list[str] | None = None, width: int = 1280, height: int = 720) -> tuple[str, dict]:
    """-> (HTML, spec). `families` defaults to a random 3-6 of FAMILIES (always including ambient half the time)."""
    r = random.Random(seed)
    fam = list(families) if families else r.sample(FAMILIES, r.randint(3, 6))
    pal = _palette(r)
    layout = r.choice(LAYOUTS)
    spec = {
        "seed": seed, "layout": layout, "families": fam, "font": r.choice(FONTS), "radius": r.choice([0, 3, 6, 10, 16]),
        "density": r.choice([0.8, 1.0, 1.2]), "palette": pal, "title": _words(r, r.randint(1, 3)),
        "pages": [{"title": _words(r, 2), "cards": [{"title": _words(r, r.randint(1, 3)), "text": _words(r, r.randint(5, 16)),
                                                       "button": _words(r, 1)} for _ in range(r.randint(3, 6))]}
                  for _ in range(r.randint(2, 4))],
        "nav": [_words(r, 1) for _ in range(r.randint(3, 6))],
        "timings": {"dialog_at": r.uniform(1, 12), "toast_s": r.uniform(0.6, 3.0), "load_s": [0.2, r.uniform(0.6, 4.0)],
                    "transition_s": r.uniform(0.15, 0.6), "shift_at": r.uniform(1.5, 10), "enable_after": r.uniform(0.5, 6),
                    "long_op_s": r.uniform(5, 15), "ambient_period": r.uniform(0.8, 3.0), "toast_every": r.uniform(4, 14),
                    "error_p": r.uniform(0.1, 0.4)},
        "dialog_kind": r.choice(["cookie", "permission", "update", "confirm", "newsletter"]),
        "toast_pos": r.choice(["top", "bottom", "top-right", "bottom-right"]),
        "width": width, "height": height,
    }
    return _html(spec), spec


def _html(s: dict) -> str:
    p = s["palette"]
    css = f"""
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:'{s['font']}',sans-serif; background:{p['bg']}; color:{p['fg']}; font-size:{round(14 * s['density'])}px; overflow:hidden; }}
header {{ display:flex; align-items:center; gap:18px; padding:10px 20px; background:{p['accent']}; color:#fff; }}
header .nav button {{ background:transparent; color:#fff; border:0; font:inherit; padding:6px 10px; cursor:pointer; }}
#clock {{ margin-left:auto; font-variant-numeric:tabular-nums; }}
main {{ position:relative; height:calc(100vh - 52px); overflow:hidden; }}
.page {{ position:absolute; inset:0; padding:{round(18 * s['density'])}px; display:grid; grid-template-columns:repeat({2 if s['layout'] in ('form','article') else 3}, 1fr); gap:16px; align-content:start; }}
.card {{ background:{p['card']}; border-radius:{s['radius']}px; padding:14px; box-shadow:0 1px 3px rgba(0,0,0,.18); min-height:90px; }}
.card h3 {{ margin:0 0 6px; font-size:1.05em; }}
.btn {{ margin-top:8px; padding:7px 14px; border-radius:{s['radius']}px; border:0; background:{p['accent']}; color:#fff; font:inherit; cursor:pointer; }}
.btn[disabled] {{ background:#aaa; color:#eee; cursor:not-allowed; }}
input {{ width:100%; padding:6px; margin-top:6px; font:inherit; border:1px solid #bbb; border-radius:{s['radius']}px; }}
.anim-in {{ animation: slidein var(--t) ease-out; }} .anim-out {{ animation: fadeout var(--t) ease-in forwards; }}
@keyframes slidein {{ from {{ transform:translateX(40px); opacity:0; }} to {{ transform:none; opacity:1; }} }}
@keyframes fadeout {{ to {{ opacity:0; }} }}
@keyframes spin {{ to {{ transform:rotate(360deg); }} }}
@keyframes pulse {{ 50% {{ opacity:.45; }} }}
.spinner {{ width:28px; height:28px; border:4px solid #ccc; border-top-color:{p['accent']}; border-radius:50%; animation:spin .8s linear infinite; margin:10px auto; }}
.skeleton {{ height:12px; background:#ccc; margin:8px 0; border-radius:4px; animation:pulse 1s infinite; }}
.overlay {{ position:fixed; inset:0; background:rgba(0,0,0,.45); display:flex; align-items:center; justify-content:center; z-index:50; }}
.dialog {{ background:{p['card']}; color:{p['fg']}; border-radius:{s['radius']}px; padding:22px; width:{360 + 40 * len(s['nav'])}px; box-shadow:0 8px 30px rgba(0,0,0,.4); }}
.toast {{ position:fixed; z-index:60; padding:12px 18px; border-radius:{s['radius']}px; color:#fff; box-shadow:0 4px 14px rgba(0,0,0,.3); }}
.toast.success {{ background:#2e9e4a; }} .toast.info {{ background:#3a6fd8; }} .toast.error {{ background:#c93a3a; }}
.banner {{ background:#fde2e2; color:#8a1f1f; border:1px solid #e8a5a5; padding:10px 14px; border-radius:{s['radius']}px; grid-column:1/-1; }}
.ad {{ grid-column:1/-1; height:{round(70 * s['density'])}px; display:flex; align-items:center; justify-content:center; color:#fff; border-radius:{s['radius']}px; }}
.progress {{ height:14px; background:#ddd; border-radius:7px; overflow:hidden; }} .progress > div {{ height:100%; background:{p['accent']}; width:0; }}
"""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{s['title']}</title><style>{css}</style></head>
<body><header><b>{s['title']}</b><div class="nav">{''.join(f'<button data-tid="nav{i}" data-nav="{i % len(s['pages'])}">{n}</button>' for i, n in enumerate(s['nav']))}</div><span id="clock"></span></header>
<main id="main"></main>
<script>
const SPEC = {json.dumps(s)};
{_JS}
</script></body></html>"""


_JS = r"""
let rng = (() => { let x = SPEC.seed * 9301 + 49297; return () => (x = (x * 9301 + 49297) % 233280) / 233280; })();
const T = SPEC.timings, F = new Set(SPEC.families);
const G = {page: 0, version: 0, animating: 0, loading: 0, long_op: false, error: false, focus: 'main'};
const bump = () => { G.version++; };
const main = document.getElementById('main');
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; }
function anim(node, cls, secs, done) {
  G.animating++; node.style.setProperty('--t', secs + 's'); node.classList.add(cls);
  setTimeout(() => { node.classList.remove(cls); G.animating--; if (done) done(); }, secs * 1000 + 30);
}
function renderPage(i, transition) {
  const old = main.querySelector('.page');
  const pg = el('div', 'page'); pg.dataset.page = i;
  const P = SPEC.pages[i];
  P.cards.forEach((c, k) => {
    const card = el('div', 'card');
    card.appendChild(el('h3', '', c.title));
    card.appendChild(el('div', '', c.text));
    if (SPEC.layout === 'form' && k % 2 === 0) { const inp = el('input'); inp.placeholder = c.title; inp.dataset.tid = `in${i}_${k}`; card.appendChild(inp); }
    const b = el('button', 'btn', c.button); b.dataset.tid = `b${i}_${k}`; b.dataset.act = ['load', 'toast', 'nav', 'dialog', 'load', 'longop'][k % 6];
    if (F.has('disabled') && k === 1) { b.disabled = true; setTimeout(() => { b.disabled = false; bump(); }, T.enable_after * 1000); }
    card.appendChild(b); pg.appendChild(card);
  });
  if (F.has('ambient')) { const ad = el('div', 'ad', 'Sponsored'); ad.dataset.ambient = 1; pg.insertBefore(ad, pg.children[Math.min(2, pg.children.length)]); }
  if (old) {
    anim(old, 'anim-out', T.transition_s, () => old.remove());
  }
  main.appendChild(pg);
  if (transition) anim(pg, 'anim-in', T.transition_s);
  G.page = i; bump();
}
function toast(kind, text, secs) {
  const t = el('div', `toast ${kind}`, text); t.dataset.toast = kind;
  const pos = SPEC.toast_pos; t.style[pos.startsWith('top') ? 'top' : 'bottom'] = '64px';
  if (pos.endsWith('right')) t.style.right = '24px'; else { t.style.left = '50%'; t.style.transform = 'translateX(-50%)'; }
  document.body.appendChild(t); bump();
  setTimeout(() => { t.remove(); bump(); }, secs * 1000);
}
function dialog() {
  if (document.querySelector('.overlay')) return;
  const texts = {cookie: 'We use cookies to improve your experience.', permission: 'This site wants to send you notifications.',
                 update: 'A new version is available. Restart now?', confirm: 'Discard unsaved changes?', newsletter: 'Subscribe to our newsletter?'};
  const o = el('div', 'overlay'); const d = el('div', 'dialog', `<h3>${SPEC.title}</h3><p>${texts[SPEC.dialog_kind]}</p>`);
  ['Accept', 'Later'].forEach((n, j) => { const b = el('button', 'btn', n); b.dataset.tid = `dlg${j}`; b.onclick = (ev) => { ev.stopPropagation(); o.remove(); G.focus = 'main'; bump(); }; d.appendChild(b); });
  o.appendChild(d); document.body.appendChild(o); G.focus = 'dialog'; bump();
}
function banner(text) {
  const pg = main.querySelector('.page'); if (!pg || pg.querySelector('.banner')) return;
  const b = el('div', 'banner', text + ' <button class="btn" data-tid="close_err">Dismiss</button>'); pg.insertBefore(b, pg.firstChild);
  b.querySelector('button').onclick = (ev) => { ev.stopPropagation(); b.remove(); G.error = false; bump(); };
  G.error = true; bump();
}
function load(card, then) {
  const ph = el('div', '', rng() < .5 ? '<div class="spinner"></div>' : '<div class="skeleton"></div><div class="skeleton" style="width:70%"></div>');
  ph.dataset.loading = 1; card.appendChild(ph); G.loading++; bump();
  const secs = T.load_s[0] + rng() * (T.load_s[1] - T.load_s[0]);
  setTimeout(() => { ph.remove(); G.loading--; then(); }, secs * 1000);
}
function longop(card) {
  if (G.long_op) return; G.long_op = true;
  const p = el('div', 'progress', '<div></div>'); const lab = el('div', '', '0%'); card.appendChild(p); card.appendChild(lab); bump();
  const t0 = performance.now(), dur = T.long_op_s * 1000;
  const tick = () => { const f = Math.min(1, (performance.now() - t0) / dur); p.firstChild.style.width = (f * 100) + '%'; lab.textContent = Math.round(f * 100) + '%';
    if (f < 1) requestAnimationFrame(tick); else { p.remove(); lab.textContent = 'Completed'; G.long_op = false; bump(); } };
  tick();
}
document.addEventListener('click', (ev) => {
  const b = ev.target.closest('button'); if (!b || b.disabled || document.querySelector('.overlay')) return;
  if (b.dataset.nav !== undefined) { renderPage(+b.dataset.nav, F.has('navigation')); return; }
  const card = b.closest('.card'); const act = b.dataset.act;
  if (act === 'dialog' && F.has('dialog')) return dialog();
  if (act === 'longop' && F.has('long_op')) return longop(card);
  if (act === 'nav') return renderPage((G.page + 1) % SPEC.pages.length, F.has('navigation'));
  const finish = () => {
    if (F.has('error') && rng() < T.error_p) { banner('Request failed: ' + SPEC.title + ' is unavailable.'); if (F.has('toast')) toast('error', 'Error: could not save', T.toast_s); return; }
    card.querySelector('div').textContent = 'Updated ' + new Date().toLocaleTimeString(); bump();
    if (F.has('toast')) toast(rng() < .5 ? 'success' : 'info', rng() < .5 ? 'Saved' : 'Changes applied', T.toast_s);
  };
  if (F.has('loading') || act === 'load') load(card, finish); else finish();
});
// timed dynamics
if (F.has('dialog')) setTimeout(dialog, T.dialog_at * 1000);
if (F.has('toast')) setInterval(() => toast('info', 'New message received', T.toast_s), T.toast_every * 1000);
if (F.has('shift')) setTimeout(() => { const pg = main.querySelector('.page'); if (!pg) return;
  const ad = el('div', 'ad', 'Limited offer'); ad.style.background = '#e08a2c'; ad.style.height = '0px'; ad.style.transition = 'height .35s';
  pg.insertBefore(ad, pg.firstChild); G.animating++; requestAnimationFrame(() => ad.style.height = Math.round(70 * SPEC.density) + 'px');
  setTimeout(() => { G.animating--; bump(); }, 400); }, T.shift_at * 1000);
if (F.has('ambient')) setInterval(() => { document.querySelectorAll('[data-ambient]').forEach(a => { a.style.background = `hsl(${Math.floor(rng() * 360)},60%,50%)`; a.textContent = 'Sponsored · ' + SPEC.nav[Math.floor(rng() * SPEC.nav.length)]; }); }, T.ambient_period * 1000);
setInterval(() => { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }, 1000);
renderPage(0, false);
window.__gt = () => {
  const box = e => { const r = e.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]; };
  const dlg = document.querySelector('.overlay .dialog'); const toasts = Array.from(document.querySelectorAll('.toast'));
  const ban = document.querySelector('.banner');
  const els = Array.from(document.querySelectorAll('[data-tid]')).filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight; })
      .map(e => [e.dataset.tid, ...box(e), !e.disabled]);
  return {page: G.page, version: G.version, animating: G.animating > 0, loading: G.loading > 0, long_op: G.long_op,
          dialog: !!dlg, dialog_box: dlg ? box(document.querySelector('.overlay')) : null,
          toast: toasts.some(t => t.dataset.toast !== 'error'), error: G.error || toasts.some(t => t.dataset.toast === 'error'),
          toast_boxes: toasts.map(box), error_box: ban ? box(ban) : null, focus: G.focus,
          ambient_boxes: Array.from(document.querySelectorAll('[data-ambient]')).map(box).concat([box(document.getElementById('clock'))]),
          elements: els};
};
"""

// Builds a per-type, per-run retrieval slide in the Excalidraw+ "Experiments" scene
// from runs/heldout400/confusion_by_run.json. Used for slide 11 (gold-sentence recall)
// and slide 12 (answer support). Run in the scene page with window.__app set and SPEC injected:
//   SPEC.P[type][baseline][level][run] = [both, Agentic only, baseline only, neither]
//   SPEC.C[type][method][level]        = questions in [3, 2, 1, 0] of the 3 runs
//   SPEC.levels (column-group labels), SPEC.frame {num, name}, SPEC.title, SPEC.subtitle,
//   SPEC.h1, SPEC.h2, SPEC.verb ("found" / "supported"), SPEC.notes
// An existing frame whose name starts with "<num> " is cleared and refilled; otherwise a new
// frame is created right after frame num-1 and later frames shift right (one slot, 3350 px).
(async (SPEC) => {
  const a = window.__app, sc = a.scene;
  let all = sc.getElementsIncludingDeleted();
  const live = all.filter(e => !e.isDeleted);
  const bump = e => ({ ...e, version: e.version + 1, versionNonce: Math.floor(Math.random() * 2 ** 31), updated: Date.now() });
  const rid = () => Math.random().toString(36).slice(2, 12) + Math.random().toString(36).slice(2, 12);
  const frames = live.filter(e => e.type === 'frame');
  const numOf = f => parseInt((f.name.match(/^(\d+)/) || [0, 0])[1]);
  let fr = frames.find(f => numOf(f) === SPEC.frame.num);
  if (!fr) {
    const prev = frames.find(f => numOf(f) === SPEC.frame.num - 1);
    const later = new Set(frames.filter(f => f.x > prev.x).map(f => f.id));
    all = all.map(e => (!e.isDeleted && (later.has(e.id) || later.has(e.frameId))) ? bump({ ...e, x: e.x + 3350 }) : e);
    fr = { ...prev, id: rid(), x: prev.x + 3350, seed: Math.floor(Math.random() * 2 ** 31), version: 1,
      versionNonce: Math.floor(Math.random() * 2 ** 31), boundElements: null, groupIds: [], index: null };
    all.push(fr);
  }
  const tmplText = live.find(e => e.type === 'text' && e.fontFamily === 2 && !e.containerId);
  const tmplLine = live.find(e => e.type === 'line' && e.points && e.points.length === 2);
  const ctx = document.createElement('canvas').getContext('2d');
  const FONTS = { 2: 'Helvetica, Arial', 7: '"Lilita One"' };
  const DARK = '#1e1e1e', GREY = '#868e96', BLUE = '#1971c2', ORANGE = '#e8590c', RULE = '#ced4da';
  const out = [];
  const base = () => ({ id: rid(), seed: Math.floor(Math.random() * 2 ** 31), version: 1,
    versionNonce: Math.floor(Math.random() * 2 ** 31), frameId: fr.id, groupIds: [], boundElements: null,
    updated: Date.now(), index: null, isDeleted: false, angle: 0, opacity: 100, locked: false, link: null });
  const text = (s, x, y, o = {}) => {
    const size = o.size || 28, fam = o.fam || 2, lines = s.split('\n');
    ctx.font = `${size}px ${FONTS[fam]}`;
    const w = Math.ceil(Math.max(...lines.map(l => ctx.measureText(l).width))) + 6;
    const align = o.align || 'left';
    const left = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
    out.push({ ...tmplText, ...base(), type: 'text', x: fr.x + left, y: fr.y + y, width: w,
      height: lines.length * size * 1.25, text: s, originalText: s, fontSize: size, fontFamily: fam,
      textAlign: align, verticalAlign: 'top', strokeColor: o.color || DARK, containerId: null,
      autoResize: true, lineHeight: 1.25 });
    return w;
  };
  const rule = (x1, x2, y) => out.push({ ...tmplLine, ...base(), type: 'line', x: fr.x + x1, y: fr.y + y,
    width: x2 - x1, height: 0, points: [[0, 0], [x2 - x1, 0]], strokeColor: RULE, strokeWidth: 2 });
  const inline = (parts, x, y, size) => { for (const [s, c] of parts) x += text(s, x, y, { size, color: c }); };

  const TYPES = ['Comparison', 'Temporal', 'Inference'];
  const BASES = ['vs RAG', 'vs HippoRAG', 'vs RAPTOR'];
  const METHODS = ['Agentic GraphRAG', 'RAG', 'HippoRAG', 'RAPTOR'];
  const NG = SPEC.levels.length;
  const L = 104, X0 = 600, GW = 3 * 216 + 2 * 18, GSTEP = GW + 56, RIGHT = X0 + (NG - 1) * GSTEP + GW;
  const gx = g => X0 + g * GSTEP;
  const COL4 = [DARK, BLUE, ORANGE, GREY], C4 = [DARK, ORANGE, ORANGE, GREY];

  text(SPEC.title, L, 78, { size: 67.8, fam: 7 });
  text(SPEC.subtitle, L, 183, { size: 36.5, color: GREY });

  let y = 290;
  text(SPEC.h1, L, y, { size: 47 });
  inline([['Each cell, out of 100 questions:  ', GREY], [`both ${SPEC.verb}`, DARK], ['  ·  ', GREY], ['Agentic only', BLUE],
    ['  ·  ', GREY], ['baseline only', ORANGE], ['  ·  ', GREY], ['neither', GREY]], L, y + 70, 31.3);
  y += 150;
  SPEC.levels.forEach((t, g) => {
    text(t, gx(g) + GW / 2, y, { size: 30, align: 'center' });
    [0, 1, 2].forEach(r => text(`Run ${r + 1}`, gx(g) + r * 234 + 108, y + 46, { size: 26, color: GREY, align: 'center' }));
  });
  y += 92; rule(L, RIGHT, y); y += 12;
  SPEC.P.forEach((byBase, ti) => {
    text(TYPES[ti], L, y, { size: 32 }); y += 50;
    byBase.forEach((byLv, bi) => {
      text(BASES[bi], L + 30, y, { size: 28, color: GREY });
      byLv.forEach((byRun, g) => byRun.forEach((cell, r) => cell.forEach((v, k) =>
        text(String(v), gx(g) + r * 234 + 27 + k * 54, y, { size: 28, color: COL4[k], align: 'center' }))));
      y += 44;
    });
    y += 8; rule(L, RIGHT, y); y += 12;
  });

  y += 50;
  text(SPEC.h2, L, y, { size: 47 });
  inline([[`Each cell, out of 100 questions:  ${SPEC.verb} in `, GREY], ['all 3 runs', DARK], ['  ·  ', GREY],
    ['2 runs', ORANGE], ['  ·  ', GREY], ['1 run', ORANGE], ['  ·  ', GREY], ['no run', GREY]], L, y + 70, 31.3);
  y += 150;
  const SUB = ['3 runs', '2 runs', '1 run', '0 runs'];
  SPEC.levels.forEach((t, g) => {
    text(t, gx(g) + GW / 2, y, { size: 30, align: 'center' });
    SUB.forEach((s, k) => text(s, gx(g) + 85 + k * 171, y + 46, { size: 26, color: GREY, align: 'center' }));
  });
  y += 92; rule(L, RIGHT, y); y += 12;
  SPEC.C.forEach((byMethod, ti) => {
    text(TYPES[ti], L, y, { size: 32 }); y += 50;
    byMethod.forEach((byLv, mi) => {
      text(METHODS[mi], L + 30, y, { size: 28, color: mi === 0 ? BLUE : GREY });
      byLv.forEach((cell, g) => cell.forEach((v, k) =>
        text(String(v), gx(g) + 85 + k * 171, y, { size: 28, color: C4[k], align: 'center' })));
      y += 44;
    });
    y += 8; rule(L, RIGHT, y); y += 12;
  });

  y += 40;
  text(SPEC.notes.join('\n'), L, y, { size: 31.3, color: GREY });
  const H = y + SPEC.notes.length * 31.3 * 1.25 + 100;

  const name = `${SPEC.frame.num} ${SPEC.frame.name}`;
  const renames = SPEC.renames || {};
  const kept = all.map(e => {
    if (e.isDeleted) return e;
    if (e.frameId === fr.id) return bump({ ...e, isDeleted: true });
    if (e.id === fr.id) return { ...bump(e), name, height: H };
    if (e.type === 'frame' && renames[e.name]) return { ...bump(e), name: renames[e.name] };
    return e;
  });
  window.__t0 = performance.now();
  a.updateScene({ elements: [...kept, ...out], captureUpdate: 'IMMEDIATELY' });
  await new Promise(r => setTimeout(r, 500));
  const now = sc.getNonDeletedElements();
  return { added: out.length, inFrame: now.filter(e => e.frameId === fr.id).length, H: Math.round(H),
    noType: sc.getElementsIncludingDeleted().filter(e => !e.type).length,
    frames: now.filter(e => e.type === 'frame').sort((p, q) => p.x - q.x).map(f => f.name.slice(0, 18) + '@' + Math.round(f.x)) };
})

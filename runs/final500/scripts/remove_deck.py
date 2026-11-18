"""Remove every element tagged recragDeck=2026-11-13-final500 from the open scene
and put the listed frames back to their heights in the backup file."""
import json, subprocess, sys
from pathlib import Path
backup = json.loads(Path(sys.argv[1]).read_text())
heights = {e["id"]: e["height"] for e in backup["elements"] if e["type"] == "frame"}
js = """() => {
  const a = window.__xapp; const h = %s;
  const next = a.scene.getElementsIncludingDeleted().map(e => {
    if (e.customData && e.customData.recragDeck === '2026-11-13-final500' && !e.isDeleted) return {...e, isDeleted: true, version: e.version + 1, updated: Date.now()};
    if (e.type === 'frame' && h[e.id] !== undefined && e.height !== h[e.id]) return {...e, height: h[e.id], version: e.version + 1, updated: Date.now()};
    return e; });
  a.updateScene({elements: next, captureUpdate: 'IMMEDIATELY'});
  return a.scene.getNonDeletedElements().filter(e => e.customData && e.customData.recragDeck === '2026-11-13-final500').length;
}""" % json.dumps(heights)
print(subprocess.run(["chrome-devtools-axi", "eval", js], capture_output=True, text=True).stdout.split("\nhelp")[0])

"""Append runs/final500/deck_ops.json to the open Experiments scene (Helium tab).

Uploads the ops in chunks to window.__ops, then appends the new elements and
sets the listed frame heights. Existing elements are never changed except the
frame heights. Prints what the scene reports afterwards.
"""

import json
import subprocess
from pathlib import Path

ops = Path("runs/final500/deck_ops.json").read_text()


def ev(js: str) -> str:
    out = subprocess.run(["chrome-devtools-axi", "eval", js], capture_output=True, text=True)
    return (out.stdout + out.stderr).split("\nhelp")[0]


print(ev("() => { window.__ops = ''; return 'reset' }"))
size = 150_000
for start in range(0, len(ops), size):
    chunk = json.dumps(ops[start : start + size])
    ev(f"() => {{ window.__ops += {chunk}; return window.__ops.length }}")
print(ev("() => window.__ops.length"), "of", len(ops))
print(
    ev(
        """() => {
  const a = window.__xapp;
  const ops = JSON.parse(window.__ops);
  const current = a.scene.getElementsIncludingDeleted();
  const have = new Set(current.map(e => e.id));
  const fresh = ops.elements.filter(e => !have.has(e.id));
  const next = current.map(e => (e.type === 'frame' && ops.frames[e.id] && !e.isDeleted)
    ? {...e, height: ops.frames[e.id], version: e.version + 1, versionNonce: Math.floor(Math.random() * 2 ** 31), updated: Date.now()}
    : e);
  a.updateScene({elements: [...next, ...fresh], captureUpdate: 'IMMEDIATELY'});
  const after = a.scene.getNonDeletedElements();
  const added = after.filter(e => e.customData && e.customData.recragDeck === '2026-11-13-final500');
  return JSON.stringify({added: added.length, total: after.length, noIndex: added.filter(e => !e.index).length});
}"""
    )
)

"""Save NQ validation rows 0-499, processed by dataset._natural_questions_record,
to runs/final500/natural_questions/records.json (read via RECRAG_NQ_RECORDS)."""
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from dataset import load_natural_questions_record
out = Path("runs/final500/natural_questions/records.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps([asdict(load_natural_questions_record(i)) for i in range(500)], ensure_ascii=False))
print(out, sha256(out.read_bytes()).hexdigest())

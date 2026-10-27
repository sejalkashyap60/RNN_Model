"""Portable local summaries of completed runs (no model arrays or corpus text)."""
import json
from pathlib import Path

def load_history(path):
    path=Path(path)
    if not path.exists(): return []
    rows=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows,list): raise ValueError("Experiment history must be a JSON list")
    return rows

def append_history(path, experiment):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=load_history(path)
    row={k:experiment[k] for k in ("id","config","corpus_hash","vocab_size","sequence_counts","metrics")}
    if experiment.get("checkpoint"): row["checkpoint"]=experiment["checkpoint"]
    rows.append(row)
    path.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    return rows

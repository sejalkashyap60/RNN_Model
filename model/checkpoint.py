import json
from pathlib import Path
import numpy as np
from .rnn import VanillaRNN
from .text import Vocabulary

def save(model, vocab, path, metadata=None):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    arrays=dict(model.params)
    gradients=getattr(model,"last_gradients",None)
    if gradients:
        arrays.update({f"grad_{name}":value for name,value in gradients.items()})
    np.savez_compressed(path/"weights.npz", **arrays)
    info={"format":1,"words":vocab.words,"hidden_size":model.H,"metadata":metadata or {}}
    (path/"metadata.json").write_text(json.dumps(info,indent=2),encoding="utf-8")

def load(path):
    path=Path(path); info=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    if info.get("format")!=1 or not isinstance(info.get("words"),list): raise ValueError("Unsupported checkpoint metadata")
    if not info["words"] or info["words"][0]!="<unk>" or len(set(info["words"]))!=len(info["words"]):
        raise ValueError("Checkpoint vocabulary is invalid")
    vocab=Vocabulary(info["words"])
    if vocab.words!=info["words"]: raise ValueError("Checkpoint vocabulary ordering is invalid")
    model=VanillaRNN(len(vocab.words),int(info["hidden_size"]))
    with np.load(path/"weights.npz",allow_pickle=False) as arrays:
        expected=set(model.params); optional={f"grad_{name}" for name in model.params}
        if not expected.issubset(arrays.files) or set(arrays.files)-expected not in (set(),optional):
            raise ValueError("Checkpoint parameter or gradient set is invalid")
        for k,p in model.params.items():
            if arrays[k].shape!=p.shape: raise ValueError(f"Invalid shape for {k}")
            p[:]=arrays[k]
        if set(arrays.files)==expected|optional:
            model.last_gradients={name:arrays[f"grad_{name}"].copy() for name in model.params}
            if any(model.last_gradients[name].shape!=model.params[name].shape for name in model.params):
                raise ValueError("Checkpoint gradient shape is invalid")
    return model,vocab,info

import numpy as np

def softmax(logits):
    if logits.ndim != 2 or logits.shape[1] != 1 or not np.all(np.isfinite(logits)):
        raise ValueError("logits must be a finite column vector")
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)

class VanillaRNN:
    """One sequence produces one next-token distribution; all math is explicit."""
    def __init__(self, vocab_size, hidden_size=32, seed=7):
        if vocab_size < 2 or hidden_size < 1: raise ValueError("invalid model dimensions")
        rng = np.random.default_rng(seed); scale = 0.01
        self.V, self.H = vocab_size, hidden_size
        self.Wxh = rng.normal(0, scale, (hidden_size, vocab_size))
        self.Whh = rng.normal(0, scale, (hidden_size, hidden_size))
        self.Why = rng.normal(0, scale, (vocab_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1)); self.by = np.zeros((vocab_size, 1))
    @property
    def params(self): return {"Wxh":self.Wxh,"Whh":self.Whh,"Why":self.Why,"bh":self.bh,"by":self.by}
    @property
    def parameter_count(self): return sum(p.size for p in self.params.values())
    def forward(self, ids, details=False):
        h = np.zeros((self.H,1)); steps=[]
        for token in ids:
            if not isinstance(token, (int, np.integer)) or not 0 <= int(token) < self.V:
                raise ValueError("token index is outside the vocabulary")
            x = np.zeros((self.V,1)); x[int(token),0]=1
            input_part=self.Wxh@x; recurrent_part=self.Whh@h
            a=input_part+recurrent_part+self.bh; new_h=np.tanh(a)
            output_part=self.Why@new_h; logits=output_part+self.by; probs=softmax(logits)
            steps.append({"token":int(token),"x":x,"h_prev":h,"input":input_part,"recurrent":recurrent_part,
                "bias":self.bh.copy(),"a":a,"h":new_h,"output":output_part,"logits":logits,"probs":probs})
            h=new_h
        if not steps: raise ValueError("input sequence is empty")
        logits=self.Why@h+self.by; probs=softmax(logits)
        return (probs, logits, h, steps) if details else (probs, logits, h)
    def loss_and_gradients(self, ids, target):
        if not isinstance(target, (int, np.integer)) or not 0 <= int(target) < self.V:
            raise ValueError("target index is outside the vocabulary")
        probs, logits, h, steps=self.forward(ids, True)
        loss=float(-np.log(max(float(probs[int(target),0]),1e-15)))
        grads={k:np.zeros_like(v) for k,v in self.params.items()}
        dy=probs.copy(); dy[int(target),0]-=1
        grads["Why"] += dy@h.T; grads["by"] += dy
        dh=self.Why.T@dy
        for s in reversed(steps):
            da=dh*(1-s["h"]**2)
            grads["Wxh"] += da@s["x"].T; grads["Whh"] += da@s["h_prev"].T; grads["bh"] += da
            dh=self.Whh.T@da
        return loss,grads
    def train_example(self, ids, target, lr, clip=5.0):
        if lr <= 0 or not np.isfinite(lr): raise ValueError("learning rate must be positive and finite")
        if clip is not None and (clip <= 0 or not np.isfinite(clip)): raise ValueError("clip threshold must be positive and finite")
        loss, grads=self.loss_and_gradients(ids,target)
        norm=np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        self.last_gradient_norm=float(norm)
        self.last_update_clipped=bool(clip is not None and norm>clip)
        if clip is not None and norm>clip:
            for g in grads.values(): g *= clip/(norm+1e-12)
        self.last_gradients={name:g.copy() for name,g in grads.items()}
        for name,g in grads.items(): self.params[name] -= lr*g
        return loss

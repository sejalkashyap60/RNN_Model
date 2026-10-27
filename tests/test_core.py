import sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.text import tokenize,Vocabulary,make_sequences,one_hot
from model.rnn import VanillaRNN,softmax
from model.checkpoint import save,load
from training.trainer import train

class CoreTests(unittest.TestCase):
    def test_text(self):
        self.assertEqual(tokenize("Hello, RNN!"),["hello","rnn"])
        v=Vocabulary(tokenize("a b a")); self.assertEqual(v.encode(["missing"]),[0])
        self.assertEqual(make_sequences([1,2,3],2),[([1,2],3)])
        self.assertEqual(one_hot(2,4).shape,(4,1))
    def test_forward_and_loss(self):
        m=VanillaRNN(5,4,3); p,o,h,steps=m.forward([1,2],True)
        self.assertAlmostEqual(float(p.sum()),1); self.assertEqual(h.shape,(4,1)); self.assertEqual(o.shape,(5,1)); self.assertEqual(len(steps),2)
        self.assertEqual(m.parameter_count,2*m.V*m.H+m.H*m.H+m.H+m.V)
        for step in steps:
            np.testing.assert_allclose(step["recurrent"],m.Whh@step["h_prev"])
            self.assertEqual(step["probs"].shape,(m.V,1))
        np.testing.assert_allclose(steps[-1]["probs"],p)
        np.testing.assert_allclose(steps[0]["logits"],m.Why@steps[0]["h"]+m.by)
        loss,g=m.loss_and_gradients([1,2],3); self.assertTrue(np.isfinite(loss)); self.assertEqual(g["Whh"].shape,(4,4))
        self.assertEqual(set(g),set(m.params))
    def test_gradient_difference(self):
        m=VanillaRNN(4,3,4); ids=[1,2]; target=3; _,g=m.loss_and_gradients(ids,target); eps=1e-5
        for name,index in [("Whh",(0,1)),("Wxh",(1,2)),("Why",(2,1)),("bh",(1,0)),("by",(2,0))]:
            p=m.params[name]; old=p[index]; p[index]=old+eps; lp=m.loss_and_gradients(ids,target)[0]; p[index]=old-eps; lm=m.loss_and_gradients(ids,target)[0]; p[index]=old
            numeric=(lp-lm)/(2*eps); self.assertLess(abs(numeric-g[name][index])/(abs(numeric)+abs(g[name][index])+1e-8),1e-4)
    def test_train_checkpoint(self):
        m=VanillaRNN(4,5,1); examples=[([1],2),([2],1)]*5
        history,_=train(m,examples,20,.1); self.assertTrue(history[-1]<history[0])
        vocab=Vocabulary(["a","b","c"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d:
            save(m,vocab,d); loaded,lv,_=load(d); self.assertEqual(loaded.parameter_count,m.parameter_count); self.assertEqual(lv.words,vocab.words)
            self.assertEqual(set(loaded.last_gradients),set(m.params))
            for name in m.params: np.testing.assert_allclose(loaded.last_gradients[name],m.last_gradients[name])

if __name__=="__main__": unittest.main()

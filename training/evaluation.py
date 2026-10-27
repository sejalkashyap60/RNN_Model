"""Held-out metrics and a transparent train-only bigram baseline."""
import math
from collections import Counter, defaultdict

def evaluate_rnn(model, examples, unknown_targets=0, total_targets=None, k=5):
    if not examples: raise ValueError("No held-out examples are available")
    targets=[]; losses=[]; hits1=hitsk=0
    for context,target in examples:
        p,_,_=model.forward(context)
        target=int(target); targets.append(target)
        losses.append(-math.log(max(float(p[target,0]),1e-15)))
        ranking=p[:,0].argsort()[::-1]
        hits1 += int(int(ranking[0])==target)
        hitsk += int(target in ranking[:min(k,model.V)])
    mean_loss=sum(losses)/len(losses)
    return {"top1_accuracy":hits1/len(examples),"top5_accuracy":hitsk/len(examples),
        "mean_cross_entropy":mean_loss,"perplexity":math.exp(min(mean_loss,709)),
        "examples":len(examples),"unknown_targets":int(unknown_targets),
        "unknown_target_rate":int(unknown_targets)/max(1,int(total_targets or len(examples)))}

def evaluate_bigram(train_examples, test_examples, vocab_size, unknown_targets=0, total_targets=None, k=5):
    if not train_examples or not test_examples: raise ValueError("Bigram evaluation requires train and held-out examples")
    counts=defaultdict(Counter); unigrams=Counter()
    for context,target in train_examples:
        counts[int(context[-1])][int(target)] += 1; unigrams[int(target)] += 1
    rankings=[]; losses=[]; hits1=hitsk=0
    for context,target in test_examples:
        target=int(target); row=counts.get(int(context[-1]),Counter())
        denominator=sum(row.values())+vocab_size
        p=(row.get(target,0)+1)/denominator
        losses.append(-math.log(p))
        ranking=sorted(range(vocab_size),key=lambda idx:(row.get(idx,0)+1,-idx),reverse=True)
        rankings.append(ranking)
        hits1 += int(ranking[0]==target)
        hitsk += int(target in ranking[:min(k,vocab_size)])
    mean_loss=sum(losses)/len(losses)
    return {"top1_accuracy":hits1/len(test_examples),"top5_accuracy":hitsk/len(test_examples),
        "mean_cross_entropy":mean_loss,"perplexity":math.exp(min(mean_loss,709)),
        "examples":len(test_examples),"unknown_targets":int(unknown_targets),
        "unknown_target_rate":int(unknown_targets)/max(1,int(total_targets or len(test_examples)))}

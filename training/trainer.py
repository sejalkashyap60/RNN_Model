import time
import math

def evaluate(model, examples):
    if not examples: raise ValueError("Evaluation split has no sequences; add text or adjust the split/context length")
    losses=[]
    for context,target in examples:
        probabilities,_,_=model.forward(context)
        losses.append(-math.log(max(float(probabilities[int(target),0]),1e-15)))
    return sum(losses)/len(losses)

def train_with_evaluation(model, train_examples, validation_examples, test_examples,
                          epochs=30, lr=0.1, clip=5.0, callback=None):
    if not train_examples: raise ValueError("Training split has no sequences")
    if not validation_examples or not test_examples: raise ValueError("Validation and test splits must each contain sequences")
    if isinstance(epochs,bool) or epochs<1: raise ValueError("epochs must be a positive integer")
    if lr<=0 or not math.isfinite(lr): raise ValueError("learning rate must be positive and finite")
    if clip is not None and (clip<=0 or not math.isfinite(clip)): raise ValueError("clip threshold must be positive and finite")
    started=time.monotonic(); train_history=[]; validation_history=[]; gradient_norm_history=[]; clipped_steps_history=[]
    for epoch in range(epochs):
        losses=[]; norms=[]; clipped=0
        for x,y in train_examples:
            losses.append(model.train_example(x,y,lr,clip))
            norms.append(model.last_gradient_norm)
            clipped += int(model.last_update_clipped)
        train_loss=sum(losses)/len(losses)
        validation_loss=evaluate(model,validation_examples)
        train_history.append(train_loss); validation_history.append(validation_loss)
        gradient_norm_history.append(sum(norms)/len(norms))
        clipped_steps_history.append(clipped)
        if callback:
            callback(epoch+1,epochs,train_loss,validation_loss,train_history.copy(),
                     validation_history.copy(),time.monotonic()-started,
                     gradient_norm_history[-1],clipped)
    duration=time.monotonic()-started
    return {"train_history":train_history,"validation_history":validation_history,
            "gradient_norm_history":gradient_norm_history,"clipped_steps_history":clipped_steps_history,
            "train_loss":evaluate(model,train_examples),
            "validation_loss":evaluate(model,validation_examples),
            "test_loss":evaluate(model,test_examples),"duration":duration}

def train(model, examples, epochs=30, lr=0.1, clip=5.0, callback=None):
    if not examples: raise ValueError("No training sequences were produced")
    if epochs<1 or lr<=0 or clip<=0: raise ValueError("epochs, learning rate and clipping threshold must be positive")
    started=time.monotonic(); history=[]
    for epoch in range(epochs):
        losses=[model.train_example(x,y,lr,clip) for x,y in examples]
        avg=sum(losses)/len(losses); history.append(avg)
        if callback: callback(epoch+1,epochs,avg,history.copy(),time.monotonic()-started)
    return history, time.monotonic()-started

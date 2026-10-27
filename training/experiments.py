"""Fairly repeatable isolated experiment runs over a deterministic token split."""
import hashlib
import time

from model.rnn import VanillaRNN
from model.text import Vocabulary, split_token_stream
from training.data import prepare_examples
from training.trainer import train_with_evaluation
from training.evaluation import evaluate_rnn, evaluate_bigram

def validate_config(config):
    required={"learning_rate","hidden_size","sequence_length","epochs","train_fraction","validation_fraction","seed","clip"}
    if set(config) != required: raise ValueError("Experiment configuration is incomplete")
    if not 0.0001 <= float(config["learning_rate"]) <= 1.0: raise ValueError("learning rate must be between 0.0001 and 1")
    if not 4 <= int(config["hidden_size"]) <= 256: raise ValueError("hidden size must be between 4 and 256")
    if not 1 <= int(config["sequence_length"]) <= 20: raise ValueError("sequence length must be between 1 and 20")
    if not 1 <= int(config["epochs"]) <= 500: raise ValueError("epochs must be between 1 and 500")
    if not 0.50 <= float(config["train_fraction"]) <= 0.85: raise ValueError("training split must be between 50% and 85%")
    if not 0.05 <= float(config["validation_fraction"]) <= 0.30: raise ValueError("validation split must be between 5% and 30%")
    if float(config["train_fraction"])+float(config["validation_fraction"]) >= 0.95: raise ValueError("leave at least 5% of tokens for held-out testing")
    if isinstance(config["seed"],bool) or int(config["seed"]) != config["seed"] or not 0 <= int(config["seed"]) <= 2**32-1:
        raise ValueError("random seed must be an integer from 0 to 4,294,967,295")
    if config["clip"] is not None and not 0.1 <= float(config["clip"]) <= 20: raise ValueError("clip threshold must be from 0.1 to 20, or disabled")

def run_experiment(tokens, config, callback=None):
    validate_config(config)
    if len(tokens) < 30: raise ValueError("Use at least 30 tokens for meaningful train, validation, and test splits")
    corpus_hash=hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:12]
    train_tokens,val_tokens,test_tokens=split_token_stream(tokens,float(config["train_fraction"]),float(config["validation_fraction"]))
    # Vocabulary is learned from training text only; held-out-only words become <unk>.
    vocab=Vocabulary(train_tokens)
    splits,unknown_counts,token_counts=prepare_examples(tokens,vocab,config)
    train_examples=splits["train"]; val_examples=splits["validation"]; test_examples=splits["test"]
    model=VanillaRNN(len(vocab),int(config["hidden_size"]),int(config["seed"]))
    metrics=train_with_evaluation(model,train_examples,val_examples,test_examples,
        epochs=int(config["epochs"]),lr=float(config["learning_rate"]),clip=config["clip"],callback=callback)
    metrics["evaluation"]=evaluate_rnn(model,test_examples,unknown_counts["test"],token_counts["test"])
    metrics["bigram_baseline"]=evaluate_bigram(train_examples,test_examples,len(vocab),unknown_counts["test"],token_counts["test"])
    return {"id":f"EXP-{time.time_ns()}","config":dict(config),"corpus_hash":corpus_hash,
        "vocab_size":len(vocab),"sequence_counts":{"train":len(train_examples),"validation":len(val_examples),"test":len(test_examples)},
        "metrics":metrics,"model":model,"vocab":vocab,
        "train_examples":train_examples,"validation_examples":val_examples,"test_examples":test_examples,
        "unknown_counts":unknown_counts,"token_counts":token_counts}

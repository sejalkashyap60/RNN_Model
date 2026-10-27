"""Deterministic train/validation/test examples shared by training and evaluation."""
from model.text import make_sequences, split_token_stream

def prepare_examples(tokens, vocab, config):
    train_tokens,val_tokens,test_tokens=split_token_stream(
        tokens,float(config["train_fraction"]),float(config["validation_fraction"]))
    train_ids=vocab.encode(train_tokens); val_ids=vocab.encode(val_tokens); test_ids=vocab.encode(test_tokens)
    length=int(config["sequence_length"])
    splits={"train":make_sequences(train_ids,length),"validation":make_sequences(val_ids,length),"test":make_sequences(test_ids,length)}
    if min(map(len,splits.values())) < 1:
        raise ValueError("A split is too short for this context length; reduce context length or change split percentages")
    return (splits,{"train":sum(i==0 for i in train_ids),"validation":sum(i==0 for i in val_ids),"test":sum(i==0 for i in test_ids)},
            {"train":len(train_ids),"validation":len(val_ids),"test":len(test_ids)})

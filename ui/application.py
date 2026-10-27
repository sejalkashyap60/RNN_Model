"""Streamlit pages for the NextWord AI learning laboratory."""
from pathlib import Path
import hashlib
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from model.checkpoint import load, save
from model.text import tokenize, Vocabulary, make_sequences
from training.data import prepare_examples
from training.evaluation import evaluate_rnn, evaluate_bigram
from training.experiment_history import append_history, load_history
from training.experiments import run_experiment
from ui.architecture import high_level_dot, cell_dot, unrolled_dot
from ui.charts import heatmap, bars

PAGES=["Home / Overview","Train RNN","RNN Architecture Explorer","Prediction Visualizer",
       "Sequence Explorer","Hidden-State Explorer","Experiment Lab","Model Inspector",
       "Evaluation & Bigram Comparison","Dataset Manager","Experiment History"]

def _corpus(st, root):
    upload=st.session_state.get("corpus_upload")
    raw=upload.getvalue().decode("utf-8",errors="replace") if upload else (root/"data/sample_corpus.txt").read_text(encoding="utf-8")
    return raw,tokenize(raw),hashlib.sha256(" ".join(tokenize(raw)).encode("utf-8")).hexdigest()[:12]

def _checkpoint_choices(st, root):
    choices={}
    if st.session_state.get("model") is not None:
        choices["Current session model"]=(st.session_state.model,st.session_state.vocab,{"metadata":st.session_state.get("checkpoint_metadata",{})})
    current=root/"saved_models/current"
    if current.joinpath("metadata.json").exists() and "Current session model" not in choices:
        try: choices["saved_models/current"]=load(current)
        except (ValueError,OSError,KeyError) as exc: st.warning(f"Current checkpoint cannot be loaded: {exc}")
    for row in reversed(st.session_state.get("experiments",[])):
        rel=row.get("checkpoint")
        if not rel: continue
        path=(root/rel).resolve()
        base=(root/"saved_models/experiments").resolve()
        if base not in path.parents or not path.joinpath("metadata.json").exists(): continue
        label=f"{row.get('id','experiment')} · H={row.get('config',{}).get('hidden_size')} · {row.get('corpus_hash')}"
        if label not in choices:
            try: choices[label]=load(path)
            except (ValueError,OSError,KeyError) as exc: st.caption(f"Skipped invalid saved run {label}: {exc}")
    return choices

def _select_model(st, root, key):
    choices=_checkpoint_choices(st,root)
    if not choices:
        st.info("Train a model or load a saved checkpoint to use this page.")
        return None
    label=st.selectbox("Select model checkpoint",list(choices),key=key)
    model,vocab,info=choices[label]
    return model,vocab,info,label

def _config_defaults():
    return {"learning_rate":0.1,"hidden_size":32,"sequence_length":3,"epochs":30,
            "train_fraction":0.70,"validation_fraction":0.15,"seed":7,"clip":5.0}

def _draw_metrics(st, metrics):
    keys=[("top1_accuracy","Top-1 accuracy"),("top5_accuracy","Top-5 accuracy"),
          ("mean_cross_entropy","Mean cross-entropy"),("perplexity","Perplexity")]
    cols=st.columns(4)
    for col,(key,label) in zip(cols,keys): col.metric(label,f"{metrics[key]:.4f}")
    st.caption(f"Evaluated examples: {metrics['examples']:,} · unknown tokens: {metrics['unknown_targets']:,} ({metrics['unknown_target_rate']:.1%})")

def _train_controls(st, root, tokens, corpus_hash, experiment_mode=False):
    st.subheader("Training configuration" if not experiment_mode else "Independent experiment configuration")
    suffix="exp" if experiment_mode else "train"
    defaults={f"lr_{suffix}":0.1,f"h_{suffix}":32,f"len_{suffix}":3,f"epochs_{suffix}":30,
        f"train_split_{suffix}":0.70,f"val_split_{suffix}":0.15,f"seed_{suffix}":7,
        f"clip_on_{suffix}":True,f"clip_val_{suffix}":5.0}
    for key,value in defaults.items(): st.session_state.setdefault(key,value)
    c1,c2,c3=st.columns(3)
    lr=c1.number_input("Learning rate",0.0001,1.0,step=0.01,format="%.4f",key=f"lr_{suffix}")
    hidden=c2.slider("Hidden-state size",4,128,step=4,key=f"h_{suffix}")
    length=c3.slider("Sequence length",1,12,key=f"len_{suffix}")
    d,e,f=st.columns(3)
    epochs=d.slider("Epochs",1,200,key=f"epochs_{suffix}")
    train_fraction=e.slider("Training split",0.50,0.85,step=0.05,key=f"train_split_{suffix}")
    val_fraction=f.slider("Validation split",0.05,0.30,step=0.05,key=f"val_split_{suffix}")
    seed=st.number_input("Random seed",0,2**32-1,step=1,key=f"seed_{suffix}")
    use_clip=st.checkbox("Enable global gradient clipping",key=f"clip_on_{suffix}")
    clip_value=st.number_input("Gradient norm threshold",0.1,20.0,step=0.1,disabled=not use_clip,key=f"clip_val_{suffix}")
    config={"learning_rate":lr,"hidden_size":hidden,"sequence_length":length,"epochs":epochs,
            "train_fraction":train_fraction,"validation_fraction":val_fraction,"seed":seed,
            "clip":clip_value if use_clip else None}
    if train_fraction+val_fraction>=0.95: st.warning("Leave at least 5% of tokens for held-out testing.")
    if st.button("Run independent experiment" if experiment_mode else "Train fresh RNN",type="primary",key="run_exp" if experiment_mode else "run_train"):
        if len(tokens)<30: st.error("Use at least 30 tokens so train, validation, and test splits can produce examples.")
        elif train_fraction+val_fraction>=0.95: st.error("Training and validation splits must leave at least 5% for the test split.")
        else:
            progress=st.progress(0); metric=st.empty(); chart=st.empty(); grad_chart=st.empty()
            def update(epoch,total,train_loss,val_loss,train_hist,val_hist,elapsed,grad_norm,clipped):
                progress.progress(epoch/total)
                metric.metric("Training loss",f"{train_loss:.4f}",f"Validation {val_loss:.4f} · epoch {epoch}/{total} · {elapsed:.1f}s")
                fig,ax=plt.subplots(figsize=(8,2.8)); ax.plot(train_hist,label="Training"); ax.plot(val_hist,label="Validation"); ax.set(xlabel="Epoch",ylabel="Cross-entropy"); ax.legend(); ax.grid(alpha=.2); chart.pyplot(fig); plt.close(fig)
                grad_chart.caption(f"Mean pre-clipping gradient norm: {grad_norm:.4f} · clipped updates this epoch: {clipped}")
            try:
                result=run_experiment(tokens,config,callback=update)
                exp_dir=root/"saved_models/experiments"/result["id"]
                metadata={"experiment_id":result["id"],"corpus_hash":corpus_hash,"config":config,
                          "metrics":result["metrics"],"sequence_counts":result["sequence_counts"]}
                save(result["model"],result["vocab"],exp_dir,metadata)
                result["checkpoint"]=(Path("saved_models")/"experiments"/result["id"]).as_posix()
                st.session_state.experiments=append_history(root/"saved_models/experiments.json",result)
                st.session_state.model=result["model"]; st.session_state.vocab=result["vocab"]
                st.session_state.checkpoint_metadata=metadata
                st.session_state.current_metrics=result["metrics"]
                st.success(f"Saved {result['id']} · final train {result['metrics']['train_loss']:.4f} · validation {result['metrics']['validation_loss']:.4f} · held-out test {result['metrics']['test_loss']:.4f}")
                st.write(f"Parameters: {result['model'].parameter_count:,} · sequence windows: {result['sequence_counts']} · training time: {result['metrics']['duration']:.2f}s")
                _draw_metrics(st,result["metrics"]["evaluation"])
                st.caption(f"Bigram baseline top-1 accuracy on the same held-out examples: {result['metrics']['bigram_baseline']['top1_accuracy']:.1%}")
            except (ValueError,OSError) as exc: st.error(f"Training or checkpoint save failed: {exc}")
    return config

def _show_forward_controls(st, model, vocab, key_prefix, show_step_controls=True):
    prompt_key=f"{key_prefix}_prompt"
    st.text_input("Input sentence",value="machine learning is",key=prompt_key)
    tokens=tokenize(st.session_state[prompt_key]); ids=vocab.encode(tokens)
    if not tokens:
        st.warning("Enter at least one token."); return None
    unknown=[token for token,index in zip(tokens,ids) if index==0]
    if unknown: st.warning("Unknown tokens map to `<unk>`: "+", ".join(unknown))
    probs,logits,hidden,steps=model.forward(ids,details=True)
    if show_step_controls:
        idx_key=f"{key_prefix}_step"
        st.session_state.setdefault(idx_key,len(tokens)-1)
        st.session_state[idx_key]=min(st.session_state[idx_key],len(tokens)-1)
        previous,next_,reset=st.columns(3)
        if previous.button("Previous Step",key=f"{key_prefix}_prev"): st.session_state[idx_key]=max(0,st.session_state[idx_key]-1)
        if next_.button("Next Step",key=f"{key_prefix}_next"): st.session_state[idx_key]=min(len(tokens)-1,st.session_state[idx_key]+1)
        if reset.button("Reset",key=f"{key_prefix}_reset"): st.session_state[idx_key]=0
        selected=st.slider("Selected time step",0,len(tokens)-1,key=idx_key)
    else: selected=len(tokens)-1
    return {"tokens":tokens,"ids":ids,"unknown":unknown,"probs":probs,"logits":logits,
            "hidden":hidden,"steps":steps,"selected":selected}

def _forward_values(st,model,vocab,result,prefix):
    if result is None:return
    tokens=result["tokens"];ids=result["ids"];steps=result["steps"];idx=result["selected"];step=steps[idx]
    st.markdown("**Text processing**")
    st.dataframe([{"time":i+1,"token":token,"vocabulary index":token_id,"unknown":token_id==0} for i,(token,token_id) in enumerate(zip(tokens,ids))])
    cols=st.columns(3);cols[0].metric("Vocabulary size",model.V);cols[1].metric("Hidden size",model.H);cols[2].metric("Trainable parameters",f"{model.parameter_count:,}")
    st.markdown(f"**Selected step {idx+1}: `{tokens[idx]}`**")
    st.write({"x_t shape":step["x"].shape,"previous state shape":step["h_prev"].shape,"input contribution shape":step["input"].shape,"recurrent contribution shape":step["recurrent"].shape,"hidden state shape":step["h"].shape,"logits shape":step["logits"].shape})
    s1,s2=st.columns(2)
    s1.pyplot(heatmap(step["h"].ravel(),f"Hidden state h_{idx+1}","Hidden unit","State"));s2.pyplot(bars(step["h"].ravel(),[str(i) for i in range(model.H)],"Hidden activations"))
    for label,key in (("One-hot x_t","x"),("Previous h_(t-1)","h_prev"),("Input contribution W_xh x_t","input"),("Recurrent contribution W_hh h_(t-1)","recurrent"),("Hidden bias b_h","bias"),("Pre-activation a_t","a"),("tanh(a_t) = h_t","h"),("Output contribution W_hy h_t","output"),("Logits o_t","logits"),("Softmax probabilities p_t","probs")):
        with st.expander(label): st.code(np.array2string(step[key].ravel(),precision=5,threshold=80))
    order=np.argsort(result["probs"][:,0])[::-1]
    top=min(10,len(order)); words=[vocab.decode(int(i)) for i in order[:top]]; values=result["probs"][order[:top],0]
    st.write(f"Final input: **{' '.join(tokens)}** → top next word: **{vocab.decode(int(order[0]))}**")
    st.caption(f"Top-{top} view only; its probability total is {float(values.sum()):.1%}, with the remaining vocabulary omitted.")
    st.pyplot(bars(values,words,"Next-word softmax probabilities"))
    with st.expander("All logits and probabilities"):
        shown=order[:100]
        st.dataframe([{"word":vocab.decode(int(i)),"logit":float(result["logits"][i,0]),"probability":float(result["probs"][i,0]),"percent":float(result["probs"][i,0])*100} for i in shown])
        st.caption(f"Showing the 100 highest probabilities (or the full vocabulary if smaller) out of {model.V} words.")

def _page_overview(st,root):
    st.title("NextWord AI")
    st.write("An interactive machine-learning lab that exposes how a Vanilla RNN turns token sequences into next-word probabilities.")
    raw,tokens,digest=_corpus(st,root)
    cols=st.columns(4);cols[0].metric("Corpus tokens",f"{len(tokens):,}");cols[1].metric("Unique words",f"{len(set(tokens)):,}");cols[2].metric("Text size",f"{len(raw):,} chars");cols[3].metric("Corpus fingerprint",digest)
    st.info("The included corpus works immediately. Upload an UTF-8 `.txt` corpus in the sidebar to explore your own text. The model learns only statistical patterns in that corpus.")
    st.subheader("Learning laboratory")
    st.markdown("Use **Train RNN** for a measured train/validation/test run, **RNN Architecture Explorer** to inspect the shared recurrent weights and actual activations, and **Experiment Lab** to compare independent configurations.")
    st.text_area("Corpus preview",raw[:1500],height=150,disabled=True)
    if st.session_state.get("model") is not None:
        m=st.session_state.model; st.success(f"Loaded model: V={m.V}, H={m.H}, parameters={m.parameter_count:,}")

def _page_train(st,root,tokens,digest):
    st.title("Train RNN")
    st.markdown("Each independent sequence starts with h₀ = 0. The ordered corpus is split before sequence windows are created; held-out test data never updates weights.")
    _train_controls(st,root,tokens,digest,False)

def _page_architecture(st,root):
    st.title("RNN Architecture Explorer")
    selected=_select_model(st,root,"arch_model")
    if selected is None:return
    model,vocab,info,_=selected
    forward=_show_forward_controls(st,model,vocab,"architecture")
    if forward is None:return
    view=st.radio("Architecture view",["High-level pipeline","Vanilla RNN cell","Unrolled sequence"],horizontal=True)
    if view=="High-level pipeline": diagram=high_level_dot()
    elif view=="Vanilla RNN cell": diagram=cell_dot(forward["tokens"][forward["selected"]])
    else: diagram=unrolled_dot(forward["tokens"],forward["selected"])
    st.graphviz_chart(diagram)
    st.caption("The dotted connections in the unrolled view all point to the same shared weight matrices. The diagram is not a set of independently trained cells.")
    component=st.selectbox("Inspect an architecture component",["x_t","W_xh","W_hh","b_h","a_t","h_t","W_hy","b_y","o_t","Softmax p_t","Prediction"])
    step=forward["steps"][forward["selected"]]
    values={"x_t":step["x"],"W_xh":model.Wxh,"W_hh":model.Whh,"b_h":model.bh,"a_t":step["a"],"h_t":step["h"],"W_hy":model.Why,"b_y":model.by,"o_t":step["logits"],"Softmax p_t":step["probs"]}
    equations={"x_t":"One-hot encoding of the current token.","W_xh":"a_t = W_xh x_t + W_hh h_(t-1) + b_h","W_hh":"The same recurrent matrix transforms h_(t-1) at every token.","b_h":"Hidden-layer bias added before tanh.","a_t":"Weighted input + recurrent contribution + hidden bias.","h_t":"h_t = tanh(a_t)","W_hy":"o_t = W_hy h_t + b_y","b_y":"Output-layer bias added to logits.","o_t":"Unnormalized scores; logits are not probabilities.","Softmax p_t":"p_t = exp(o_t - max(o_t)) / sum(exp(o_t - max(o_t)))","Prediction":"Highest-scoring vocabulary item from the model distribution."}
    st.subheader(component); st.write("**Purpose:** "+equations[component]); st.write("**Equation / operation:** "+equations[component])
    if component=="Prediction": st.write(f"**Actual output:** {vocab.decode(int(np.argmax(step['probs'][:,0])))} · {float(np.max(step['probs'])):.2%}")
    else:
        arr=values[component]; st.write("**Actual tensor shape:**",arr.shape)
        if arr.ndim==2 and min(arr.shape)>1: st.pyplot(heatmap(arr,f"Actual {component}"))
        with st.expander("Numerical values"): st.code(np.array2string(arr,precision=5,threshold=200))
    st.caption("Alternative to clicking graph nodes: choose a named node above. The diagram itself is static; selected numerical panels are computed from the live forward pass.")

def _page_prediction(st,root):
    st.title("Prediction Visualizer")
    selected=_select_model(st,root,"prediction_model")
    if selected is None:return
    model,vocab,info,label=selected
    def append_best():
        text=st.session_state.get("prediction_prompt","")
        ids=vocab.encode(tokenize(text))
        if ids:
            probs,_,_=model.forward(ids); word=vocab.decode(int(np.argmax(probs[:,0])))
            st.session_state["prediction_prompt"]=f"{text.strip()} {word}".strip()
    st.button("Append predicted word and continue",on_click=append_best,key="append_word")
    result=_show_forward_controls(st,model,vocab,"prediction",False)
    if result is None:return
    st.subheader("Prediction Autopsy")
    st.write({"checkpoint":label,"checkpoint id":info.get("metadata",{}).get("experiment_id","saved model"),"configuration":info.get("metadata",{}).get("config",{}),"input tokens":result["tokens"],"vocabulary indices":result["ids"]})
    st.caption("These numerical calculations describe the model's computation; they are not a complete semantic explanation of why a word was selected.")
    _forward_values(st,model,vocab,result,"prediction_details")

def _page_sequence(st,root):
    st.title("Sequence Explorer")
    selected=_select_model(st,root,"sequence_model")
    if selected is None:return
    model,vocab,_,_=selected
    result=_show_forward_controls(st,model,vocab,"sequence")
    if result is None:return
    st.graphviz_chart(unrolled_dot(result["tokens"],result["selected"]))
    hidden=np.vstack([s["h"].ravel() for s in result["steps"]])
    st.pyplot(heatmap(hidden,"Token-by-hidden-unit activation heatmap","Hidden unit","Time step"))
    unit=st.slider("Inspect hidden unit",0,model.H-1,0,key="sequence_unit")
    st.pyplot(bars(hidden[:,unit],[f"{i+1}: {t}" for i,t in enumerate(result["tokens"])],f"Hidden unit {unit} across tokens",horizontal=False))
    if len(hidden)>1:
        st.pyplot(heatmap(np.diff(hidden,axis=0),"Hidden-state change h_t - h_(t-1)","Hidden unit","Transition"))
    st.subheader("Per-step next-word distributions")
    for i,s in enumerate(result["steps"]):
        rank=np.argsort(s["probs"][:,0])[::-1][:3]
        st.write(f"t={i+1} `{result['tokens'][i]}` → "+", ".join(f"{vocab.decode(int(j))} ({s['probs'][j,0]:.1%})" for j in rank))

def _page_hidden(st,root):
    st.title("Hidden-State Memory Explorer")
    selected=_select_model(st,root,"memory_model")
    if selected is None:return
    model,vocab,_,label=selected
    a,b=st.columns(2); a.text_input("Original sentence", "machine learning is useful",key="memory_a"); b.text_input("Modified sentence", "machine learning feels useful",key="memory_b")
    results=[]
    for key in ("memory_a","memory_b"):
        tokens=tokenize(st.session_state[key]); ids=vocab.encode(tokens)
        if not ids: st.warning("Both inputs need at least one token."); return
        results.append((tokens,model.forward(ids,True)))
    for col,(tokens,forward) in zip((a,b),results):
        probs,logits,h,steps=forward
        col.caption("Tokens: "+" → ".join(tokens)); col.pyplot(heatmap(np.vstack([s["h"].ravel() for s in steps]),"Hidden states","Unit","Time"))
        col.markdown(f"**Final prediction:** {vocab.decode(int(np.argmax(probs[:,0])))}")
        col.write([(vocab.decode(int(i)),float(probs[i,0])) for i in np.argsort(probs[:,0])[::-1][:5]])
    p1=results[0][1][0][:,0];p2=results[1][1][0][:,0]
    probability_delta=p2-p1; logit_delta=results[1][1][1][:,0]-results[0][1][1][:,0]
    changed=np.argsort(np.abs(probability_delta))[::-1][:min(20,model.V)]
    st.pyplot(bars(probability_delta[changed],[vocab.decode(int(i)) for i in changed],"Largest probability changes: modified minus original"))
    changed_logits=np.argsort(np.abs(logit_delta))[::-1][:min(20,model.V)]
    st.pyplot(bars(logit_delta[changed_logits],[vocab.decode(int(i)) for i in changed_logits],"Largest logit changes: modified minus original"))
    unit=st.slider("Compare a hidden unit",0,model.H-1,0,key="memory_unit")
    st.pyplot(bars([results[0][1][2][unit,0],results[1][1][2][unit,0]],["Original","Modified"],f"Unit {unit}: final state",horizontal=False))
    st.caption("Hidden units do not have assumed semantic meanings. Vanilla RNNs can lose information over long sequences as repeated tanh derivatives shrink gradients.")

def _page_lab(st,root,tokens,digest):
    st.title("What-If Experiment Simulator")
    st.write("Every run creates fresh parameters, uses training-only vocabulary construction, and uses the same deterministic corpus split for matching split settings.")
    config=_train_controls(st,root,tokens,digest,True)
    rows=[r for r in st.session_state.experiments if r.get("corpus_hash")==digest
        and r.get("config",{}).get("train_fraction")==config["train_fraction"]
        and r.get("config",{}).get("validation_fraction")==config["validation_fraction"]
        and r.get("metrics",{}).get("validation_history")]
    if not rows: st.info("No experiments yet for this corpus."); return
    st.subheader("Loss curves for this corpus and fixed split")
    st.caption("Curves are measurements from the saved independent runs. A larger hidden layer, longer context, or higher learning rate does not guarantee better held-out results.")
    fig,ax=plt.subplots(figsize=(10,4))
    for r in rows[-6:]:
        ax.plot(r["metrics"]["validation_history"],label=f"{r['id'][-8:]} H{r['config']['hidden_size']} lr{r['config']['learning_rate']}")
    ax.set(xlabel="Epoch",ylabel="Validation cross-entropy");ax.legend(fontsize=8);ax.grid(alpha=.2);st.pyplot(fig);plt.close(fig)
    st.dataframe([{"Run":r["id"],"LR":r["config"]["learning_rate"],"Hidden":r["config"]["hidden_size"],"Length":r["config"]["sequence_length"],"Epochs":r["config"]["epochs"],"Train":r["metrics"].get("train_loss"),"Validation":r["metrics"].get("validation_loss"),"Test":r["metrics"].get("test_loss"),"Top-1":r["metrics"].get("evaluation",{}).get("top1_accuracy"),"Checkpoint":r.get("checkpoint","not saved")} for r in rows])
    st.subheader("Compare predictions with one shared prompt")
    prompt=st.text_input("Comparison prompt","machine learning is",key="lab_prediction_prompt")
    candidates=[r for r in rows if r.get("checkpoint")][-5:]
    comparisons=[]
    for row in candidates:
        try:
            model,vocab,_=load(root/row["checkpoint"]); ids=vocab.encode(tokenize(prompt))
            if ids:
                p,_,_=model.forward(ids); idx=np.argsort(p[:,0])[::-1][0]
                comparisons.append({"Run":row["id"],"Top word":vocab.decode(int(idx)),"Probability":float(p[idx,0]),"Validation loss":row["metrics"]["validation_loss"]})
        except (ValueError,OSError,KeyError): continue
    if comparisons: st.dataframe(comparisons)
    else: st.caption("Save at least one experiment checkpoint to compare predictions.")

def _page_inspector(st,root):
    st.title("Model Inspector")
    selected=_select_model(st,root,"inspector_model")
    if selected is None:return
    model,vocab,info,label=selected
    metrics=st.columns(4);metrics[0].metric("Vocabulary V",model.V);metrics[1].metric("Hidden H",model.H);metrics[2].metric("Parameters",f"{model.parameter_count:,}");metrics[3].metric("Checkpoint",str(info.get("format","?")))
    st.latex(r"P=2VH+H^2+H+V")
    dimensions={"x_t":(model.V,1),"W_xh":model.Wxh.shape,"W_hh":model.Whh.shape,"b_h":model.bh.shape,"h_t":(model.H,1),"W_hy":model.Why.shape,"b_y":model.by.shape,"o_t":(model.V,1),"p_t":(model.V,1)}
    st.dataframe([{"Tensor":name,"Actual shape":str(shape)} for name,shape in dimensions.items()])
    gradient_map=getattr(model,"last_gradients",None)
    source=st.radio("Inspect values",["Parameters","Last applied gradients"] if gradient_map else ["Parameters"],horizontal=True)
    values=model.params if source=="Parameters" else gradient_map
    parameter=st.selectbox("Parameter matrix or bias",list(values))
    arr=values[parameter];st.pyplot(heatmap(arr,f"{source}: {parameter}"))
    with st.expander("Full numerical values"): st.code(np.array2string(arr,precision=5,threshold=500))
    st.write("Checkpoint metadata",info.get("metadata",{}))

def _page_evaluation(st,root,tokens,digest):
    st.title("Evaluation & Bigram Comparison")
    selected=_select_model(st,root,"evaluation_model")
    if selected is None:return
    model,vocab,info,label=selected;meta=info.get("metadata",{})
    if meta.get("corpus_hash")!=digest:
        st.warning("This checkpoint does not have a matching corpus fingerprint. Evaluation is disabled to prevent scoring it against a different dataset.")
        return
    config=meta.get("config")
    if not config:
        st.warning("This older checkpoint does not contain a saved split configuration. Retrain to create evaluation-ready metadata.");return
    try:
        splits,unknown,token_counts=prepare_examples(tokens,vocab,config)
        rnn_metrics=evaluate_rnn(model,splits["test"],unknown["test"],token_counts["test"])
        bigram_metrics=evaluate_bigram(splits["train"],splits["test"],model.V,unknown["test"],token_counts["test"])
    except ValueError as exc: st.error(str(exc));return
    st.caption("All metrics use the ordered held-out test split. Test examples are never used to update weights.")
    left,right=st.columns(2)
    with left: st.subheader("Vanilla RNN");_draw_metrics(st,rnn_metrics)
    with right: st.subheader("Add-one-smoothed bigram");_draw_metrics(st,bigram_metrics)
    st.subheader("Inspect a labelled held-out example")
    index=st.slider("Test example",0,len(splits["test"])-1,0,key="heldout_example")
    context,target=splits["test"][index];p,logits,h,steps=model.forward(context,True);pred=int(np.argmax(p[:,0]))
    actual=[vocab.decode(int(i)) for i in context];target_word=vocab.decode(int(target));pred_word=vocab.decode(pred)
    st.write({"context":" ".join(actual),"actual target":target_word,"predicted":pred_word,"correct":pred==int(target),"probability of actual target":float(p[int(target),0])})

def _page_dataset(st,root,tokens,digest):
    st.title("Dataset Manager")
    raw,tokens,digest=_corpus(st,root)
    seq_len=st.slider("Sequence length for examples",1,12,3,key="dataset_seq")
    train_fraction=st.slider("Training token percentage",0.50,0.85,0.70,step=0.05,key="dataset_train_split")
    val_fraction=st.slider("Validation token percentage",0.05,0.30,0.15,step=0.05,key="dataset_val_split")
    if train_fraction+val_fraction>=0.95: st.error("Leave at least 5% of the corpus for testing.");return
    train_end=int(len(tokens)*train_fraction);val_end=train_end+int(len(tokens)*val_fraction)
    preview=[{"position":i,"token":word} for i,word in enumerate(tokens[:300])]
    st.caption(f"{len(raw):,} characters · {len(tokens):,} tokens · corpus fingerprint {digest}")
    with st.expander("Tokenization preview"): st.dataframe(preview)
    frequency=Counter(tokens)
    top=frequency.most_common(20)
    if top: st.pyplot(bars([n for _,n in top],[w for w,_ in top],"20 most frequent tokens"))
    segments={"train":tokens[:train_end],"validation":tokens[train_end:val_end],"test":tokens[val_end:]}
    vocab=Vocabulary(segments["train"])
    examples={name:make_sequences(vocab.encode(items),seq_len) for name,items in segments.items()}
    st.write({"split token counts":{k:len(v) for k,v in segments.items()},"train-only vocabulary size":len(vocab),"unknown held-out tokens":{"validation":sum(v==0 for v in vocab.encode(segments["validation"])),"test":sum(v==0 for v in vocab.encode(segments["test"]))},"sequence counts":{k:len(v) for k,v in examples.items()}})
    with st.expander("Example input-target pairs"):
        name=st.selectbox("Split",list(examples),key="dataset_examples_split")
        st.dataframe([{"context":" ".join(vocab.decode(i) for i in x),"target":vocab.decode(y)} for x,y in examples[name][:20]])
    st.caption("The vocabulary is learned from training text only. Validation/test-only tokens map to <unk>; windows are made inside each split, preventing cross-boundary context leakage.")

def _page_history(st,root,digest):
    st.title("Experiment History")
    rows=st.session_state.get("experiments",[])
    if not rows: st.info("No saved experiment summaries yet.");return
    st.dataframe([{"Run":r["id"],"Corpus":r.get("corpus_hash"),"Seed":r["config"].get("seed"),"LR":r["config"].get("learning_rate"),"Hidden":r["config"].get("hidden_size"),"Length":r["config"].get("sequence_length"),"Epochs":r["config"].get("epochs"),"Train loss":r["metrics"].get("train_loss"),"Validation loss":r["metrics"].get("validation_loss"),"Test loss":r["metrics"].get("test_loss"),"Checkpoint":r.get("checkpoint","not saved")} for r in rows])
    st.caption("Summaries and checkpoints are local runtime files and may be lost in cloud restarts. No corpus text is saved in the experiment history.")
    eligible=[r for r in rows if r.get("corpus_hash")==digest]
    if eligible:
        row=st.selectbox("Inspect a run on the active corpus",eligible,format_func=lambda r:r["id"],key="history_pick")
        st.json({"config":row["config"],"sequence_counts":row["sequence_counts"],"metrics":row["metrics"],"checkpoint":row.get("checkpoint")})
        st.button("Load this configuration into Experiment Lab",key="load_history_config",
            on_click=_load_config_into_lab,args=(row["config"],))

def _load_config_into_lab(config):
    keys={"learning_rate":"lr_exp","hidden_size":"h_exp","sequence_length":"len_exp","epochs":"epochs_exp",
          "train_fraction":"train_split_exp","validation_fraction":"val_split_exp","seed":"seed_exp","clip":"clip_val_exp"}
    for source,target in keys.items():
        if source in config and config[source] is not None: st.session_state[target]=config[source]
    st.session_state["clip_on_exp"]=config.get("clip") is not None
    st.session_state["main_navigation"]="Experiment Lab"

def run_app():
    root=Path(__file__).resolve().parents[1]
    st.set_page_config(page_title="NextWord AI",page_icon="🧠",layout="wide")
    st.markdown("<style>.stApp{background:#f7f8fc}.block-container{max-width:1360px;padding-top:1.5rem}h1{letter-spacing:-.035em}</style>",unsafe_allow_html=True)
    st.title("NextWord AI")
    st.caption("Interactive Vanilla RNN visualization and experiment laboratory")
    st.session_state.setdefault("model",None);st.session_state.setdefault("vocab",None)
    if "experiments" not in st.session_state:
        try:st.session_state.experiments=load_history(root/"saved_models/experiments.json")
        except (ValueError,OSError) as exc:st.session_state.experiments=[];st.sidebar.warning(f"Experiment history could not be loaded: {exc}")
    if st.session_state.model is None and (root/"saved_models/current/metadata.json").exists():
        try:
            st.session_state.model,st.session_state.vocab,info=load(root/"saved_models/current")
            st.session_state.checkpoint_metadata=info.get("metadata",{})
        except (ValueError,OSError,KeyError) as exc:st.sidebar.error(f"Saved checkpoint is invalid: {exc}")
    st.sidebar.file_uploader("Corpus (.txt, optional)",type=["txt"],key="corpus_upload")
    page=st.sidebar.radio("Navigate",PAGES,key="main_navigation")
    raw,tokens,digest=_corpus(st,root)
    if len(raw)>2_000_000:st.error("Corpus is larger than the 2 MB application limit.");return
    if not tokens:st.error("Corpus is empty after tokenization.");return
    if page==PAGES[0]:_page_overview(st,root)
    elif page==PAGES[1]:_page_train(st,root,tokens,digest)
    elif page==PAGES[2]:_page_architecture(st,root)
    elif page==PAGES[3]:_page_prediction(st,root)
    elif page==PAGES[4]:_page_sequence(st,root)
    elif page==PAGES[5]:_page_hidden(st,root)
    elif page==PAGES[6]:_page_lab(st,root,tokens,digest)
    elif page==PAGES[7]:_page_inspector(st,root)
    elif page==PAGES[8]:_page_evaluation(st,root,tokens,digest)
    elif page==PAGES[9]:_page_dataset(st,root,tokens,digest)
    elif page==PAGES[10]:_page_history(st,root,digest)

def high_level_dot():
    return '''digraph G {rankdir=LR; bgcolor="transparent"; node [shape=box,style="rounded,filled",fillcolor="#e9f4ff",color="#3c6eaa",fontname="Arial"];
    input [label="Input text"]; tok [label="Tokenization"]; ids [label="Vocabulary\nindices"]; hot [label="One-hot\nx_t"]; rnn [label="Vanilla RNN\nh_t = tanh(Wxh x + Whh h + bh)",fillcolor="#e8e4ff"];
    out [label="Output layer\no = Why h + by"]; soft [label="Stable softmax\np = softmax(o)",fillcolor="#e5faef"]; pred [label="Ranked next word"];
    input -> tok -> ids -> hot -> rnn -> out -> soft -> pred;}'''

def cell_dot(token=None):
    token_label=f"Input x_t\\n{token}" if token else "Input x_t"
    return f'''digraph G {{rankdir=LR; bgcolor="transparent"; nodesep=.42; node [shape=box,style="rounded,filled",fontname="Arial",color="#405b87"];
    xt [label="{token_label}",fillcolor="#dff2ff"]; wxh [label="W_xh",fillcolor="#fff0d4"]; hprev [label="Previous state h_(t-1)",fillcolor="#e8e4ff"];
    whh [label="Shared recurrent weights W_hh",fillcolor="#fff0d4"]; sum [label="Sum + b_h",shape=circle,fillcolor="#f3f4f6"]; tanh [label="tanh",shape=ellipse,fillcolor="#e5faef"];
    ht [label="Current hidden h_t",fillcolor="#e8e4ff"]; why [label="W_hy",fillcolor="#fff0d4"]; logits [label="Logits o_t",fillcolor="#dff2ff"];
    soft [label="Softmax p_t",fillcolor="#e5faef"]; pred [label="Next-word ranking",fillcolor="#dff2ff"];
    xt -> wxh -> sum; hprev -> whh -> sum; sum -> tanh -> ht; ht -> why -> logits -> soft -> pred;
    by [label="b_y"]; by -> logits; bh [label="b_h"]; bh -> sum;}}'''

def unrolled_dot(tokens, selected=0, window=8):
    if not tokens: return high_level_dot()
    start=max(0,min(selected-window//2,len(tokens)-window)); end=min(len(tokens),start+window)
    lines=['digraph G {rankdir=LR; bgcolor="transparent"; node [shape=box,style="rounded,filled",fontname="Arial"];']
    lines.append('weights [label="Shared W_xh, W_hh, W_hy",shape=note,fillcolor="#fff0d4"];')
    for i in range(start,end):
        color="#e8e4ff" if i==selected else "#e9f4ff"
        lines.append(f'x{i} [label="x_{i+1}\\n{tokens[i]}",fillcolor="{color}"]; h{i} [label="h_{i+1}",fillcolor="{color}"]; x{i} -> h{i} [label="W_xh, b_h + tanh"];')
        lines.append(f'weights -> h{i} [style=dashed,color="#9a85dc"];')
        if i==start: lines.append(f'h0{i} [label="h_{i}",fillcolor="#f5efff"]; h0{i} -> h{i} [label="W_hh"];')
        else: lines.append(f'h{i-1} -> h{i} [label="W_hh"];')
    lines.append('}')
    return "\n".join(lines)

# RNN mathematics

For `V` vocabulary items and `H` hidden units, `x_t` is one-hot `(V,1)`. `W_xh` is `(H,V)`, `W_hh` `(H,H)`, `W_hy` `(V,H)`, `b_h` `(H,1)`, and `b_y` `(V,1)`.

The recurrent transition is `a_t = W_xh x_t + W_hh h_(t-1) + b_h`, then `h_t=tanh(a_t)`. The final state produces `o=W_hy h_T+b_y`. Stable softmax is `exp(o-max(o))/sum(exp(o-max(o)))`. For target `q`, loss is `-log(p_q)`.

The logits gradient is `dy=p-one_hot(q)`. Hence `dW_hy=dy h_T^T`, `db_y=dy`, and the initial hidden derivative is `W_hy^T dy`. At each reversed time step, `da_t=dh_t elementwise (1-h_t²)`, `dW_xh += da_t x_t^T`, `dW_hh += da_t h_(t-1)^T`, `db_h += da_t`, and `dh_(t-1)=W_hh^T da_t`. Global norm clipping rescales all gradients when their combined Euclidean norm exceeds the configured threshold. SGD subtracts learning-rate times each clipped gradient.

The dashboard reports the average pre-clipping global gradient norm and the number of clipped example updates per epoch. Checkpoints can include the final clipped gradients actually applied in that run; Model Inspector labels these as gradients, separately from parameter values.

Finite differences perturb one parameter by `+epsilon` and `-epsilon`, estimate a central derivative, and compare it with BPTT. This catches indexing and chain-rule errors in the analytical implementation.

## Evaluation protocol

The token stream is split in its original order into train, validation, and held-out test segments before context windows are formed. No generated window crosses a boundary. Each run updates weights only using training windows. Validation cross-entropy is measured after each epoch; test cross-entropy is measured after all epochs and is not used for optimization or model selection. This ordered split suits the next-word task but can make a small corpus's test estimate noisy.

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Frozen-MLP causal tracing
#
# This notebook executes causal traces with all the MLP modules for a token disabled, by freezing them at the corrupted state.

# +
import os, re
import torch, numpy
import importlib, copy
import transformers
from collections import defaultdict
from util import nethook
from matplotlib import pyplot as plt
from experiments.causal_trace import ModelAndTokenizer, make_inputs, predict_from_input, decode_tokens
from experiments.causal_trace import layername, find_token_range, trace_with_patch, plot_trace_heatmap

mt = ModelAndTokenizer('gpt2-xl')


# -

# ## Tracing a single location
#
# The strategy here is to use three interventions, rather than two:
#
# 1. As before, corrupt a subset of the input.
# 2. As before, restore a subset of the internal hidden states to see
#    which ones restore the output.
# 3. But now, while doing so, freeze a set of MLP modules when processing
#    the specific subject token, so that they are stuck in the corrupted
#    state.  This reveals effect of the hidden states on everything
#    except for those particular MLP executions.
#    
# This three-way intervention is implemented in `trace_with_repatch`

def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == "transformer.wte":
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            ["transformer.wte"] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


# ## Tracing all locations
#
# Now we just need to repeat it over all locations, and draw the heatmaps.

# +
def calculate_hidden_flow_3(
    mt, prompt, subject, token_range=None, samples=10, noise=0.1, window=10, disable_mlp=False, disable_attn=False
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == 'last_subject':
        token_range = [e_range[1] - 1]
    low_score = trace_with_patch(mt.model, inp, [], answer_t, e_range,
            noise=noise).item()
    differences = trace_important_states_3(
        mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise,
        disable_mlp=disable_mlp, disable_attn=disable_attn, token_range=token_range
    )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind="",
    )

def trace_important_states_3(model, num_layers, inp, e_range, answer_t, noise=0.1, disable_mlp=False, disable_attn=False, token_range=None):
    ntoks = inp["input_ids"].shape[1]
    table = []
    zero_mlps = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        zero_mlps = []
        if disable_mlp:
            zero_mlps = [(tnum, layername(L, 'mlp')) for L in range(0, num_layers)]
        if disable_attn:
            zero_mlps += [(tnum, layername(L, 'attn')) for L in range(0, num_layers)]
        row = []
        for layer in range(0, num_layers):
            r = trace_with_repatch(
                model,
                inp,
                [(tnum, layername(layer))],
                zero_mlps, # states_to_unpatch
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)




# +
prefix = 'Megan Rapinoe plays the sport of'
entity = 'Megan Rapinoe'

no_mlp_r = calculate_hidden_flow_3(mt, prefix, entity, disable_mlp=True)
plot_trace_heatmap(no_mlp_r, title='Impact with MLP at last subject token disabled')
ordinary_r = calculate_hidden_flow_3(mt, prefix, entity)
plot_trace_heatmap(ordinary_r, title='Impact with MLP enabled as usual')


# -

# ## Comparing the with-MLP and without-MLP traces
#
# Plotting on a bar graph makes it easier to see the difference between the causal effects with and without MLP enabled.

# +
def plot_last_subject(mt, prefix, entity, token_range='last_subject', savepdf=None):
    ordinary, no_mlp = calculate_last_subject(mt, prefix, entity, token_range=token_range)
    plot_comparison(ordinary, no_mlp, prefix, savepdf=savepdf)

def calculate_last_subject(mt, prefix, entity, cache=None, token_range='last_subject'):
    def load_from_cache(filename):
        try:
            dat = numpy.load(f'{cache}/{filename}')
            return {k: v if not isinstance(v, numpy.ndarray)
                   else str(v) if v.dtype.type is numpy.str_
                   else torch.from_numpy(v)
                   for k, v in dat.items()}
        except FileNotFoundError as e:
            return None
    no_mlp_r = load_from_cache('no_mlp.npz')
    ordinary_r = load_from_cache('orindary.npz')
    if no_mlp_r is None or ordinary_r is None:
        no_mlp_r = calculate_hidden_flow_3(mt, prefix, entity, disable_mlp=True, token_range=token_range)
        ordinary_r = calculate_hidden_flow_3(mt, prefix, entity, token_range=token_range)
        if cache is not None:
            os.makedirs(cache, exist_ok=True)
            for r, filename in [(no_mlp_r, 'no_mlp.npz'), (ordinary_r, 'orindary.npz')]:
                numpy.savez(f'{cache}/{filename}',
                            **{k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in r.items()})
    return ordinary_r['scores'][0] - ordinary_r['low_score'], no_mlp_r['scores'][0] - ordinary_r['low_score']

def plot_comparison(ordinary, no_mlp, title, savepdf=None):
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        import matplotlib.ticker as mtick
        fig, ax = plt.subplots(1, figsize=(6, 1.5), dpi=300)
        ax.bar([i - 0.2 for i in range(48)], ordinary, width=0.4, color='#7261ab', label='Impact of single state on P')
        ax.bar([i + 0.2 for i in range(48)], no_mlp, width=0.4, color='#f3701b', label='Impact with MLP disabled')
        ax.set_title(title) #'Impact of individual hidden state at last subject token with MLP disabled')
        ax.set_ylabel('Average Causal Effect')
        #ax.set_xlabel('Layer at which the single hidden state is restored')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylim(None, max(0.025, ordinary.max() * 1.05))
        ax.legend()
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

if False: # Some representative cases.
    plot_last_subject(mt, "Megan Rapinoe plays the sport of", "Megan Rapinoe")
    plot_last_subject(mt, "The Big Bang Theory premires on", "The Big Bang Theory")
    plot_last_subject(mt, "Germaine Greer's domain of work is", "Germaine Greer")
    plot_last_subject(mt, "Brian de Palma works in the area of", "Brian de Palma")
    plot_last_subject(mt, "The headquarter of Zillow is in downtown", "Zillow")
    plot_last_subject(mt, "Mitsubishi Electric started in the 1900s as a small company in", "Mitsubishi")
    plot_last_subject(mt, "Mitsubishi Electric started in the 1900s as a small company in", "Mitsubishi Electric")
    plot_last_subject(mt, "Madame de Montesson died in the city of", "Madame")
    plot_last_subject(mt, "Madame de Montesson died in the city of", "Madame de Montesson")
    plot_last_subject(mt, "Edmund Neupert, performing on the", "Edmund Neupert")

# -

plot_last_subject(mt, "Megan Rapinoe plays the sport of", "Megan Rapinoe")


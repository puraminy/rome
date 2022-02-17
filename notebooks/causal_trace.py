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

# <a href="https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/causal_trace.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" align="left"/></a>&nbsp;or in a local notebook.

# + language="bash"
# !(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit
# cd /content && rm -rf /content/rome
# git clone https://github.com/kmeng01/rome rome > install.log 2>&1
# ln -s rome/counterfact .
# pip install -r /content/rome/scripts/colab_reqs/rome.txt >> install.log 2>&1
# -

IS_COLAB = False
try:
    import google.colab, torch, sys
    if not torch.cuda.is_available():
        print("Change runtime type to include a GPU.")
    sys.path.append('/content/rome')
    IS_COLAB = True
except:
    pass

# # Causal Tracing
#
# A demonstration of the double-intervention causal tracing method.
#
# The strategy used by causal tracing is to understand important
# states within a transfomer by doing two interventions simultaneously:
#
# 1. Corrupt a subset of the input.  In our paper, we corrupt the subject tokens
#    to frustrate the ability of the transformer to accurately complete factual
#    prompts about the subject.
# 2. Restore a subset of the internal hidden states.  In our paper, we scan
#    hidden states at all layers and all tokens, searching for individual states
#    that carry the necessary information for the transformer to recover its
#    capability to complete the factual prompt.
#
# The traces of decisive states can be shown on a heatmap.  This notebook
# demonstrates the code for conducting causal traces and creating these heatmaps.

# %load_ext autoreload
# %autoreload 2

# The `experiments.causal_trace` module contains a set of functions for running causal traces.
#
# In this notebook, we reproduce, demonstrate and discuss the interesting functions.
#
# We begin by importing several utility functions that deal with tokens and transformer models.

import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from experiments.causal_trace import ModelAndTokenizer, layername, guess_subject, plot_trace_heatmap
from experiments.causal_trace import make_inputs, decode_tokens, find_token_range
from experiments.causal_trace import predict_token, predict_from_input

# Now we load a model and tokenizer, and show that it can complete a couple factual statements correctly.

torch.set_grad_enabled(False)
mt = ModelAndTokenizer('gpt2-xl', low_cpu_mem_usage=IS_COLAB)
predict_token(mt, ['Shaquille O\'Neal plays the sport of',
               'Megan Rapinoe plays the sport of'
              ], return_p=True)


# ## Tracing a single location
#
# The core intervention in causal tracing is captured in this function:
#
# `trace_with_patch` a single causal trace.
#
# It enables running a batch of inferences with two interventions.
#
#   1. Random noise can be added to corrupt the inputs of some of the batch.
#   2. At any point, clean non-noised state can be copied over from an
#      uncorrupted batch member to other batch members.
#   
# The convention used by this function is that the zeroth element of the
# batch is the uncorrupted run, and the subsequent elements of the batch
# are the corrupted runs.  The argument tokens_to_mix specifies an
# be corrupted by adding Gaussian noise to the embedding for the batch
# inputs other than the first element in the batch.  Alternately,
# subsequent runs could be corrupted by simply providing different
# input tokens via the passed input batch.
#
# To ensure that corrupted behavior is representative, in practice, we
# will actually run several (ten) corrupted runs in the same batch,
# each with its own sample of noise.
#
# Then when running, a specified set of hidden states will be uncorrupted
# by restoring their values to the same vector that they had in the
# zeroth uncorrupted run.  This set of hidden states is listed in
# states_to_patch, by listing [(token_index, layername), ...] pairs.
# To trace the effect of just a single state, this can be just a single
# token/layer pair.  To trace the effect of restoring a set of states,
# any number of token indices and layers can be listed.
#
# Note that this function is also in experiments.causal_trace; the code
# is shown here to show the logic.

def trace_with_patch(
    model,            # The model
    inp,              # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,        # Answer probabilities to collect
    tokens_to_mix,    # Range of tokens to corrupt (begin, end)
    noise=0.1,        # Level of noise to add
    trace_layers=None # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

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
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        ["transformer.wte"] +
            list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2)
        return probs, all_traced

    return probs


# ## Scanning all locations
#
# A causal flow heatmap is created by repeating `trace_with_patch` at every individual hidden state, and measuring the impact of restoring state at each location.
#
# The `calculate_hidden_flow` function does this loop.  It handles both the case of restoring a single hidden state, and also restoring MLP or attention states.  Because MLP and attention make small residual contributions, to observe a causal effect in those cases, we need to restore several layers of contributions at once, which is done by `trace_important_window`.

# +
def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
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
    low_score = trace_with_patch(mt.model, inp, [], answer_t, e_range,
            noise=noise).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
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
        kind=kind or "",
    )

def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


# -

# ## Plotting the results
#
# The `plot_trace_heatmap` function draws the data on a heatmap.  That function is not shown here; it is in `experiments.causal_trace`.
#

# +
def plot_hidden_flow(
    mt, prompt, subject=None, samples=10, noise=0.1, window=10, kind=None, savepdf=None
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf)
    
def plot_all_flow(mt, prompt, subject=None):
    for kind in [None, "mlp", "attn"]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# -

# The following prompt can be changed to any factual statement to trace.

plot_all_flow(mt, 'Megan Rapinoe plays the sport of')

# Here we trace a few more factual statements from a file of test cases.

with open('counterfact/compiled/known_1000.json') as f:
    knowns = json.load(f)
for knowledge in knowns[:5]:
    plot_all_flow(mt, knowledge['prompt'], knowledge['subject'])

# Evaluation

Two steps: convert a training checkpoint to Hugging Face format, then score it.

```bash
# 1. Convert (CPU only). Writes eval/model_checkpoints/<run>/<step>/
python eval/checkpoint_converter.py runs/small_32k/step10000

# 2. Evaluate (one GPU). Prints one JSON line per eval.
python eval/eval_suite.py eval/model_checkpoints/small_32k/step10000 blimp,arc_easy,hellaswag,piqa,dolma_bpb
```

All scores are byte-normalized (bits per byte, BPB), so models with different
tokenizers are compared on the same footing.

`dolma_bpb` scores held-out documents from the training file. It reproduces the
1% split of `tokenize_data.py` (seed 42), so it needs the raw data to be a single
uncompressed `.jsonl` file at the path it had when tokenized.

`eval_suite.py` also contains a `nemotron_bpb` eval and a loader for
`trainer_mixture` checkpoints. Both belong to later experiments and are not
needed here.

## Mean BPB margin

BLiMP, ARC-Easy, HellaSwag, and PIQA report `mean_bpb_margin` alongside accuracy:

```text
example margin = mean(BPB of incorrect options) - BPB of correct option
mean_bpb_margin = mean(example margins)
```

Higher is better; units are bits per byte. Incorrect options have equal weight within each example, and examples have equal weight within each dataset, even when answer lengths or option counts differ. For BLiMP, each pair's margin is `BPB(bad) - BPB(good)`. Multiple-choice tasks score answers conditioned on their existing prompts.

A positive margin means the gold answer beats the average distractor; it can still lose to an individual distractor. The existing candidate scores supply the margin with no additional model forward passes.

## BLiMP

Evaluates all 67 subsets of `nyu-mll/blimp` (67,000 grammatical/ungrammatical pairs).

For fully cached datasets, set `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` when submitting to avoid Hub requests. Offline BLiMP discovery reads the local dataset cache and requires all 67 subsets.

Each sentence is scored independently, using BOS (or EOS) as starting context. All sentence tokens are scored; context and padding are excluded. No terminal EOS is added. Sentences exceeding the context limit raise an error.

```text
BPB(sentence) = summed negative log probability / (ln(2) × UTF-8 byte count)
```

A pair is correct if `BPB(good) < BPB(bad)`; otherwise it is wrong, including ties.

Reports:

- `accuracy`: correct pairs / total pairs (0–1).
- `correct`: number of correct pairs.
- `pairs`: number of evaluated pairs.
- `mean_bpb_margin`: mean of `BPB(bad) - BPB(good)`.

Every pair has equal weight. This custom accuracy uses BPB; standard BLiMP compares total sentence probabilities without byte normalization.

## ARC-Easy

Evaluates all 2,376 questions in the `test` split of `allenai/ai2_arc`, configuration `ARC-Easy`, with no demonstrations or answer generation.

For each question, use `Question: <question>\nAnswer:` as the prompt and score each candidate answer separately. Prompt and answer are tokenized separately; a leading space is included in each answer's scored tokens and UTF-8 byte count. Only answer tokens contribute to BPB; BOS/EOS starting context, prompt, and padding are excluded from the loss.

Choose the lowest-BPB answer (first listed choice breaks ties) and compare its label with `answerKey`. Reports `accuracy` (correct / total), `correct`, `total` questions, and `mean_bpb_margin`.

The complete prompt plus answer and starting token must fit the context limit; otherwise evaluation raises an error. This is custom byte-normalized accuracy, not the standard ARC-Easy scoring rule.

## HellaSwag

Evaluates a random sample of 1,000 examples from the labeled `validation` split of `Rowan/hellaswag` (shuffle seed 42, without replacement). Loads/caches the full split before sampling; the same sample is used for every model. Uses the activity label and scenario as context, following [harness prompt formatting and bracket cleanup](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/utils.py). No demonstrations or generation.

Score only each of the four endings, using the same tokenization, leading-space, byte-count, and context-limit rules as ARC-Easy. Choose the lowest-BPB ending; the first listed choice breaks ties.

Reports `accuracy` (correct / total), `correct`, `total`, and `mean_bpb_margin`. This is custom byte-normalized accuracy.

## Dolma held-out BPB

```bash
python eval_suite.py MODEL_CHECKPOINT dolma_bpb
```

Evaluates a uniform random sample of **1,000 nonempty held-out documents** from the converted checkpoint's training source (`dolma3_100gb.jsonl` for the current runs). Reproduces training's 1% split with seed 42, then uses a separate seed-42 reservoir sampler without replacement. One initial source scan per data source builds a cache in `.cache/dolma_samples/`; models sharing that source reuse it. The result records the data file used. For a checkpoint without conversion metadata, the fallback source is `../data/dolma3_250gb.jsonl`.

Each document starts with BOS (or EOS). Long documents use overlapping windows with stride half the context limit (256 for these models); every text token is scored exactly once. Documents do not share context. Starting tokens and terminal EOS are not scored. Empty documents are excluded before sampling.

Reports `bpb = total text NLL / (ln(2) × total UTF-8 bytes)`, plus `documents`, `bytes`, `tokens`, and `empty_documents`. Lower BPB is better. This is corpus BPB, not an average of document BPBs. It uses the same held-out text across tokenizers, but different context windows from training's packed validation loss.

## PIQA

Evaluates 1,000 randomly sampled validation examples from `baber/piqa` (shuffle seed 42, without replacement). Loads/caches the full split before sampling.

Prompt: `Question: <goal>\nAnswer:`. Score the two solutions using the same answer-only BPB, leading-space, and context-limit rules as ARC-Easy. Choose the lower-BPB solution; the first choice wins ties. No demonstrations or generation.

Reports `accuracy` (correct / total), `correct`, `total`, and `mean_bpb_margin`. This is custom byte-normalized accuracy.


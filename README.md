# SFT with LoRA on Qwen2.5-0.5B

Supervised fine-tuning of `Qwen/Qwen2.5-0.5B` on `HuggingFaceH4/ultrachat_200k`,
comparing LoRA at ranks {1, 4, 16} against full-weight fine-tuning.

| File | Role |
|---|---|
| `data.py` | UltraChat loading, chat-template formatting, **assistant-only loss masking**, padding collator |
| `train.py` | One training run (LoRA at a given rank, or full fine-tuning); records params / memory / time / loss curves |
| `generate.py` | Qualitative base-vs-fine-tuned generations on held-out prompts |
| `report.py` | Loss-curve figure + summary table across runs |
| `run_all.sh` | Runs all four configurations, then the report and generations |

## Run

```bash
pip install -r requirements.txt
bash run_all.sh
```

Outputs land in `runs/<config>/metrics.json` and `results/`
(`loss_curves.png`, `summary.md`, `generations.md`).

Single run:

```bash
python train.py --mode lora --rank 4 --out runs/lora_r4
python train.py --mode full          --out runs/full
```

On a single 24 GB GPU the whole sweep takes roughly 30–60 minutes. Reduce
`--max-len` or `--batch-size` if you hit OOM (keep them equal across runs, or the
memory and time comparison is meaningless).

## Design decisions worth stating in the write-up

**Loss is computed on assistant tokens only.** Each conversation is rendered with
Qwen's chat template; user turns, the system turn and all `<|im_start|>role`
headers are masked to `-100`. The closing `<|im_end|>` of each assistant turn *is*
supervised, which is what teaches the model to stop. Chat templates are
prefix-consistent, so the assistant span is recovered exactly as the difference
between `apply_chat_template(messages[:i], add_generation_prompt=True)` and
`apply_chat_template(messages[:i+1])`; `data.py` asserts this rather than
assuming it.

**Everything except the parameterization is held fixed.** Same 1000 training
conversations (fixed shuffle seed), same 100 held-out `test_sft` conversations,
same 3 epochs, effective batch 16, lr 1e-4 with 3% warmup + cosine decay, same
max length 1024, same seed, same evaluation schedule. The only thing that varies
between runs is LoRA rank, or LoRA vs. full weights.

**LoRA `alpha = 2r`.** This keeps the LoRA scaling factor `alpha/r` constant
across ranks, so a rank difference isn't confounded with an effective
learning-rate difference on the adapter. Pass `--lora-alpha` to override.

**Same learning rate for full fine-tuning.** The assignment asks for comparable
settings, so full FT uses lr 1e-4 too. That is higher than one would normally
pick for full-weight tuning, so `run_all.sh` has a commented-out extra run at
1e-5 — worth including as a footnote so it's clear full FT isn't being
handicapped by the shared hyperparameter.

**Validation loss is token-weighted.** Per-batch means are re-weighted by the
number of supervised tokens, so the reported number is a true average over the
held-out set rather than an average of per-batch averages.

**Memory numbers.** Weights are held in fp32 with bf16 autocast in every run, so
`torch.cuda.max_memory_allocated` differences reflect the real difference:
gradients and Adam moments for all 494M parameters vs. only the adapter. Peak
memory is measured after `reset_peak_memory_stats()`, immediately before the
training loop. Note that activations dominate at rank 1 vs. 16, so the *peak
memory* gap between LoRA ranks is small — the large gap is LoRA vs. full.

## Expected shape of the results

Trainable parameters with `target_modules="all-linear"` on Qwen2.5-0.5B
(q/k/v/o + gate/up/down across 24 layers; `lm_head` is excluded by PEFT):

| Config | Trainable params | ≈ % of 494M |
|---|---:|---:|
| LoRA r=1 | ~0.54 M | ~0.11% |
| LoRA r=4 | ~2.2 M | ~0.44% |
| LoRA r=16 | ~8.7 M | ~1.7% |
| Full | 494 M | 100% |

`train.py` prints the exact counts; use those, not these estimates.

Qualitatively, the base `Qwen2.5-0.5B` is a *base* model, not the Instruct
variant — given a chat-formatted prompt it typically rambles, continues the
prompt, invents further turns, or fails to emit a stop token. All fine-tuned
variants should produce a single, terminated, conversational reply. That
contrast is the main thing the generation samples need to show; differences
*between* ranks will be much subtler than the difference from the base model.

## Pointer for the LaTeX solution

> The experiment is implemented in `sft_lora/`: `train.py` runs a single
> configuration (`--mode lora --rank {1,4,16}` or `--mode full`), `data.py`
> builds the UltraChat SFT dataset with assistant-only loss masking, `report.py`
> produces the table and loss curves, and `generate.py` produces the qualitative
> comparison. `run_all.sh` reproduces every number reported here.

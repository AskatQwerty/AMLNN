# Follow-up evidence: 26 November 2026

| Folder | What it contains | Scope |
|---|---|---|
| [resolution/full_validation](resolution/full_validation/) | Four prediction files, metrics and exact image IDs | All 5,000 validation images |
| [resolution/pilot_100](resolution/pilot_100/) | Earlier small resolution comparison | Fixed 100-image subset; historical pilot |
| [resolution/teddy_bears](resolution/teddy_bears/) | Detector boxes, scores and generated captions | One illustrative COCO photo, image 113403 |
| [finetuning/run](finetuning/run/) | Configuration, history, vocabulary, summary and epoch-0 predictions | Training stopped partway through epoch 1 |
| [finetuning/validation_100](finetuning/validation_100/) | Paired baseline/partial-checkpoint predictions and metrics | Same fixed 100-image subset |

The larger-resolution modes scored worse on full validation. Partial fine-tuning increased BLEU-4 but decreased CIDEr-D on the small subset. Neither result establishes an improvement over the released model. See the [analysis](../../docs/FOLLOWUP_EXPERIMENTS.md).

Run this command from the repository root to reproduce all follow-up caption metrics on CPU:

```powershell
python tools/recompute_followup_metrics.py
```

It needs the captioning Python requirements, but no model weights, CUDA, COCO JPEGs or training. It scores the saved captions and all human references using the original project tokenizer and `pycocoevalcap==1.2`. The files include exact image IDs. Scores from 100 and 5,000 images must not be ranked against each other.

`source_manifest.json` records exact copies of the local evidence. Absolute paths in the original manifests are provenance, not required installation paths. Personal photo examples and duplicate copies of the same detector figures were omitted.

The partial checkpoint is available in the [follow-up release](https://github.com/AskatQwerty/AMLNN/releases/tag/followup-experiments-2026-11-26); it processed 22,688 training images and 709 optimizer steps. `last.pt` resumes that run; `best.pt` is still the epoch-0 baseline. No follow-up test evaluation was performed.

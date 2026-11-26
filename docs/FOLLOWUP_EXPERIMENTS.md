# Follow-up experiments: November 26, 2026

These experiments extend the original submission. They do not replace its selected frozen fusion model, archived scores or documents. No new test-set results are reported.

## What changed, and what did not

We tested two ideas:

1. Increase image resolution at inference, keeping the released weights unchanged.
2. Partially fine-tune both CNNs together with the decoder.

The first experiment finished on all 5,000 validation images. The second stopped during its first training epoch; only a paired 100-image diagnostic is available for the partial checkpoint. The recommended model remains the original epoch 11 checkpoint with ResNet input 224×224, detector short side 800, beam size 5 and length penalty 0.

## Higher resolution: full validation

The four modes use the same selected checkpoint, all 5,000 validation images, all original human references, the project tokenizer and the same beam-search settings. CIDEr-D uses the native `pycocoevalcap==1.2` scale; BLEU is in 0–1.

| ResNet input | Detector resize, short/maximum long side | CIDEr-D | BLEU-1 | BLEU-4 |
|---|---|---:|---:|---:|
| **224×224** |**800 / 1333**|**0.980339**|**0.741423**|**0.316192**|
| 224×224 |1200 / 2000|0.967319|0.738930|0.313161|
| 448×448 |800 / 1333|0.951277|0.734776|0.311360|
| 448×448 |1200 / 2000|0.931831|0.729297|0.305034|

Relative to the rerun baseline, CIDEr-D decreased by approximately 1.33%, 2.96% and 4.95%, respectively. Every enlarged-input mode also decreased BLEU-1 and BLEU-4.

This rerun's baseline is 0.980339. The original submitted selected score is 0.9801; retain both as separately recorded evaluations. Small numerical differences between inference paths, batches and kernels are possible. The resolution comparison should use its own matched baseline.

**Conclusion:** raising resolution at inference alone did not improve this checkpoint on full validation. A larger input changes the visual-feature distribution learned by the decoder. This experiment does not establish whether a model trained or adapted at the new resolution would improve.

At 448×448, ResNet50 produces a larger feature map, which is pooled back to 7×7 before the decoder. The model still receives 49 spatial tokens. This was not an experiment with 196 decoder memory positions or a different backbone downsampling schedule.

Saved evidence: [full validation](../results/followup_20261126/resolution/full_validation/).

### Why the 100-image pilot was misleading

An earlier fixed 100-image pilot gave CIDEr-D 1.0214 for both enlarged inputs versus 0.9981 for the original inputs. The larger full comparison reversed that apparent gain.

Both results are kept for traceability. The pilot guided a hypothesis; it did not establish a final improvement. CIDEr-D uses reference-corpus statistics, so subset values should not be ranked directly against full-validation values.

Saved evidence: [100-image pilot](../results/followup_20261126/resolution/pilot_100/).

### The three teddy bears

On COCO validation image 113403, the original detector already retained three distinct high-confidence bear boxes. The caption nevertheless described two bears. Therefore, this example does not support the claim that the detector found only two objects.

Increasing resolution did not reliably fix the description: the generated captions still used an incorrect count or omitted it. Detecting regions and correctly using them during language generation are different tasks. This is a qualitative case, not an estimate of counting accuracy.

Saved image and diagnosis: [three-bear evidence](../results/followup_20261126/resolution/teddy_bears/).

For an illustrated explanation of resizing, detector boxes and captions, see the [visual guide](VISUAL_EXAMPLES.md) and [CPU example notebook](../Data_Visual_Examples.ipynb).

### Timing interpretation

All four evaluations together took about 3 hours 39 minutes on the local laptop. Individual measured times are preserved in the result files. They are not controlled architecture speed benchmarks: sequential execution, GPU clocks, power state and warm-up can affect the comparison. In particular, the measurements do not demonstrate that larger images are intrinsically faster.

## Joint CNN fine-tuning: partial run

The notebook starts from the selected frozen fusion model. It updates the decoder and a limited set of CNN components:

| Component | Fine-tuning behavior |
|---|---|
| Caption decoder and fusion parameters | Trainable |
| ResNet50 final block, layer4 | Trainable |
| Faster R-CNN final backbone block, FPN and RoI box head | Trainable |
| Early backbone blocks and BatchNorm state | Fixed |
| Detector RPN and final predictor | Fixed |

The microbatch is 4 images, with gradient accumulation over 8 steps for an effective batch of 32. Learning rates are 5e-5 for the decoder and 1e-5 for the CNNs. The implementation uses AMP and stores resumable checkpoints.

The saved `last.pt` is **partway through epoch 1**:

- 22,688 of 113,287 training images processed.
- 709 successful optimizer steps.
- 0 recorded skipped AMP optimizer steps.
- Best evaluated checkpoint remains epoch 0, before this fine-tuning.

This is an unfinished experiment. The epoch-0 best checkpoint must not be presented as a trained fine-tuning winner.

### Paired check on 100 validation images

The initialization and partial `last.pt` were evaluated on the same fixed 100 validation IDs, with all references, beam size 5, length penalty 0 and maximum generation length 33. Each evaluation loaded its complete encoder and decoder state. Validation loss used one fixed caption per image and was averaged over 1,226 non-padding target tokens.

| State | Token loss | CIDEr-D | BLEU-1 | BLEU-4 |
|---|---:|---:|---:|---:|
| Initialization, epoch 0 |2.037349|0.998056|0.729331|0.314499|
| Partial epoch 1 |2.038020|0.987387|0.728335|0.321285|

CIDEr-D fell by approximately 1.07%, while BLEU-4 rose by approximately 2.16%. Loss rose slightly and BLEU-1 fell slightly. These are mixed signals on a small subset; they do not justify replacing the original model.

The 100-image diagnostic did not update the weights or replace the run's full-validation history. No full-validation score for the partial epoch-1 checkpoint is reported here. Its next meaningful assessment is completion of an epoch followed by the unchanged full 5,000-image validation protocol.

Saved evidence: [run metadata](../results/followup_20261126/finetuning/run/) and [paired evaluation](../results/followup_20261126/finetuning/validation_100/).

## Run or resume

- [Resolution notebook](../FinalFusion_Resolution.ipynb): expensive validation launch is off by default.
- [Fine-tuning notebook](../FinalFusion_FineTune.ipynb): training launch is off by default; five sequential training cells save progress.
- [Reproduction guide](REPRODUCING.md): environment, dataset, prepared-caption restoration and checkpoint paths.
- [November 26 release](https://github.com/AskatQwerty/AMLNN/releases/tag/followup-experiments-2026-11-26): `prepared_caption_data.zip` and `partial_finetune_resume.zip`.

The original release remains the supported inference package. The follow-up resume archive is experimental and contains a partial training state.

## Limits and next decision

These follow-ups use one seed and do not provide confidence intervals. The 100-image check is exploratory; the resolution experiment is a full-validation comparison but does not include adaptation at the new resolution. Neither changes the original detector-pretraining caveat: the local caption test is drawn from COCO train2017, also the documented source of detector pretraining.

Keep the original released model for inference. Continue fine-tuning only as an experiment, select checkpoints on full validation, and compare any candidate against the unchanged baseline under the same generation settings.

# Image Captioning: 79 completed models and spatial attention

**All 79 global-vector experiments are complete.** They include the original 75-model grid and four ResNet50 depth comparisons. B's 25 recurrent runs have been collected on A; no further work on B is required.

`FinalAML.ipynb` and `experiments/all_encoders_v5_100epochs/` preserve those results. The next stage is the separate, self-contained **`SpatialAttention.ipynb`**: it analyzes the saved models and adds two spatial-attention experiments. Their full training is pending. The new notebook reads the original 79 runs without changing their notebooks, checkpoints or predictions.

## Saved controls and new models

The controls are the best validation configurations in each relevant family:

| Saved global-vector control | Layers | Validation CIDEr-D | BLEU-4 |
|---|---:|---:|---:|
| ResNet50 + LSTM | 3 | 0.8081028 | 0.2514701 |
| ResNet50 + image-prefix Transformer | 3 | 0.8578049 | 0.2632592 |

Four layers lowered CIDEr-D to 0.7670167 for LSTM and 0.8387274 for Transformer. The three-layer LSTM's improvement over two layers was only 0.0034192 CIDEr-D. These are single-seed results; small differences do not establish a reliable architectural advantage.

The two new models keep three layers, width/embedding 512 and dropout 0.3:

- **LSTM with additive spatial attention:** a learned projection maps 49 image locations to 512 dimensions. The model processes their mean as an initial visual step, then attends to the locations using its previous hidden state at each word.
- **Transformer with cross-attention:** three `TransformerDecoderLayer` blocks use causal text queries and the 49 projected visual locations as memory. Text has learned positions; memory has learned row/column positions. The model keeps 8 heads, feed-forward width 2048 and pre-normalization.

The new models train from scratch. They have different parameter counts and cannot load the old global-vector checkpoints. Compare them with the matched saved controls using the same validation protocol.

## Open the new notebook

Start Jupyter in the project folder with the working `deep-learning` CUDA kernel. All required code and imports are inside `SpatialAttention.ipynb`; it does not execute the old notebook or require separate Python helper files. Its default switches are:

```python
EXTRACT_SPATIAL_FEATURES = False
RUN_PREFLIGHT = False
RUN_SPATIAL_TRAINING = False
```

Run All in this review mode to analyze the 79 saved runs and inspect cache status. If the spatial cache is missing or incomplete, set the existing `EXTRACT_SPATIAL_FEATURES` flag to `True` and run the extraction cell. Extraction resumes saved progress and reuses completed sources. Once both source caches are complete, return that flag to `False`, set `RUN_SPATIAL_TRAINING=True`, and run the training section or Run All. This starts or resumes only the two new models.

Each new experiment must pass a batch/backward/generation check and temporary small-set memorization before full training. The diagnostic model has dropout disabled, allows up to 2,000 steps and is discarded before the main model is created. `RUN_PREFLIGHT=True` can run these checks explicitly; leaving it False does not bypass the required preflight inside a new training run.

## Spatial cache

Global average pooling removed spatial information, so the old vectors cannot recover the 49 locations. The new cache uses the **same frozen ResNet50 ImageNet V2 weights, 224 x 224 resize + padding, ImageNet normalization and CUDA AMP float16 computation** as the global cache. Features are taken after layer4 and before average pooling: `[2048,7,7]` becomes `[49,2048]` in row-major order.

The cache lives in `cache/features/resnet50_spatial_v1/`. It stores **123,287 images x 49 x 2,048 values as float16**, approximately **23.045 GiB**, plus small metadata/ID files. Sources are 118,287 `train2017` images and 5,000 `val2017` images. The source train file also supplies the separately held-out labeled test rows; extraction uses no test captions or scores. Official unlabeled `test2017` is excluded.

Recreating the cache requires the original COCO JPEGs; an annotations-only transfer folder is insufficient. Once the spatial cache exists, decoder training uses it directly. Each source has explicit image-ID order, metadata and durable progress; completed batches are flushed before progress advances. Spatial means are checked against the saved global vectors within floating-point tolerance. Old global feature files are not overwritten. Allow additional disk space for the new checkpoints and any transfer archives.

## Training protocol, saving and analysis

Training uses Adam 0.001, batch size 64, maximum 100 epochs, patience 3 on validation CIDEr-D, gradient clipping 5.0 and seed 42. CNN weights stay frozen. Both models use the existing 113,287 training images, 5,000 validation images, training-only vocabulary and caption preprocessing. Each training image receives one random human caption per epoch. Validation loss uses a fixed caption; metrics retain all original references. The 5,000-image labeled test remains unused during model selection.

Keep `pycocoevalcap==1.2` and the saved project tokenizer. BLEU-1/BLEU-4 use the 0–1 scale; CIDEr-D uses the implementation's native scale, including its factor 10. These local scores use different tokenization from the official COCO leaderboard.

All new outputs go to **`experiments/resnet50_spatial_v1/`**. Analysis tables/plots are saved in its `analysis/` subfolder. Each spatial run saves `best.pt`/`last.pt` with model, optimizer, AMP scaler, RNG and early-stopping state; `best_weights.pt`/`last_weights.pt` contain plain weights. Configuration, vocabulary, history, summary and validation predictions are retained. Resume with identical settings; an interrupted run continues from its last committed checkpoint.

Timers record extraction and training work separately. Active training time includes validation and checkpoint writing, while pauses and preflight are excluded. Device information is saved. Different GPUs and stopping epochs affect total times, so totals are not isolated architecture-speed measurements. If the agreed batch size does not fit, use a separately named result group for a changed protocol.

The new notebook compares saved global controls with pending or completed spatial runs and distinguishes CNN activation maps from learned decoder attention. Its verification section displays an available `spatial_verification.json` report; a development check is not a completed training experiment. Only claim the checks and runs recorded in the actual report/results.

Local development checks passed on the RTX 4060 Laptop GPU: both spatial models passed batch-64 forward/backward, attention/causal/PAD checks, greedy generation, AMP train/eval, four-image memorization, and exact model/optimizer/scaler/CPU+CUDA RNG checkpoint reload. A temporary Transformer run with 8 training and 4 validation images also resumed with exactly matching weights after interruption. The complete spatial cache was extracted and every row checked for finite, nonzero features; recorded active extraction time is about 12m54s. Full spatial model training has **not** been run. The LSTM has 18,657,044 trainable parameters and the Transformer has 23,940,884, compared with 17,606,932 and 20,778,772 in their saved global controls.

## Preserve and share

Keep all 79 original result folders, the original notebook, prepared captions, split IDs, vocabulary and global feature caches. The spatial stage uses a separate cache and result group. B's collected results and existing archive remain valid; do not retrain or re-export them unless another transfer is needed.

Large datasets, caches, checkpoints and archives stay out of GitHub and are transferred separately. A machine receiving only the old training-data archive still needs the new spatial cache or the original JPEGs to create it. For reproduction details, retain `requirements-captioning.txt` and `environment-reference.json`; use the existing working CUDA environment.

The branch remains [codex/50-experiments](https://github.com/AskatQwerty/AMLNN/tree/codex/50-experiments). After the spatial stage, the two self-contained notebooks can be combined into the single final submission, preserving distinct model names and output groups. Select finalists on validation and reserve the labeled test until that choice is fixed.

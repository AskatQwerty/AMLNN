# Source helpers

The four project notebooks are self-contained. These files preserve their development sources for inspection.

- `final_fusion/models.py`: the trained fusion decoder architecture.
- `final_fusion/runtime.py`: datasets, training, resume and token losses.
- `final_fusion/region_cache.py` and `expand_region_cache.py`: detector cache extraction and full-train expansion.
- `stage5_handoff/`: prepared-data reader, shared metric tokenizer, frozen region extraction and attention-LSTM definitions.
- `fusion_decoding_search_20261115/`: greedy/beam search and validation search.
- `fusion_beam_followup_20261115/`: shared evaluation utilities imported by the search module.

The sibling directory layout is intentional: the original modules import each other using these paths. Historical CLI defaults point to the original Windows project and can include expired scheduling times. Adapt those constants before running a CLI. The notebook is the recommended starting point. `inference.py` is a notebook fragment and expects the region-helper definitions in the same namespace. No imports or training jobs were executed during packaging.

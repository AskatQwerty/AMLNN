"""Batched beam decoding for the unchanged FinalFusion model."""
import math

import torch


@torch.no_grad()
def fusion_beam_search(
    model, grid_features, region_features, start_id, end_id, max_steps=33,
    region_mask=None, region_boxes=None, global_features=None,
    beam_size=3, length_penalty=0.6,
):
    """Return token lists without START/EOS/PAD, using frozen model weights.

    Keep up to K live non-EOS prefixes and K completed captions per image.
    Live prefixes at the same step are ranked by cumulative log probability.
    Completed/final candidates use score / ((5 + L) / 6)**alpha, where L
    excludes START and includes EOS when emitted. At the length limit, compare
    both completed captions and unfinished prefixes using that same formula.
    Beam size 1 deliberately uses the model's exact greedy implementation.
    """
    if type(beam_size) is not int or beam_size < 1:
        raise ValueError("beam_size must be a positive integer.")
    if type(max_steps) is not int or not 1 <= max_steps < model.max_positions:
        raise ValueError("max_steps must fit the word positions after the image prefix.")
    if not isinstance(length_penalty, (int, float)) or not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError("length_penalty must be finite and nonnegative.")
    special = (model.pad_id, start_id, end_id)
    if any(type(value) is not int or value < 0 for value in special) or len(set(special)) != 3:
        raise ValueError("PAD, START and END must have distinct nonnegative integer IDs.")
    was_training = model.training
    model.eval()
    try:
        if beam_size == 1:
            return model.greedy(
                grid_features, region_features, start_id, end_id, max_steps=max_steps,
                region_mask=region_mask, region_boxes=region_boxes, global_features=global_features,
            )
        memory, memory_padding, image_token = model.prepare_memory(
            grid_features, region_features, region_mask, region_boxes, global_features,
        )
        batch_size = len(memory)
        if batch_size == 0:
            return []
        device = memory.device
        prefixes = torch.full((batch_size, 1, 1), start_id, dtype=torch.long, device=device)
        live_scores = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        finished_scores = torch.full((batch_size, beam_size), -torch.inf, device=device)
        finished_tokens = torch.full((batch_size, beam_size, max_steps + 1), model.pad_id,
                                     dtype=torch.long, device=device)
        finished_lengths = torch.zeros(batch_size, beam_size, dtype=torch.long, device=device)
        current_beams = 0
        legal_tokens = None
        live_length = 0

        for length in range(1, max_steps + 1):
            beams = prefixes.shape[1]
            if beams != current_beams:
                # Every beam for one image sees identical visual memory/masks.
                beam_memory = memory.repeat_interleave(beams, dim=0)
                beam_padding = memory_padding.repeat_interleave(beams, dim=0)
                beam_image = image_token.repeat_interleave(beams, dim=0)
                current_beams = beams
            logits = model.decode_memory(
                beam_memory, beam_padding, beam_image, prefixes.reshape(batch_size * beams, -1),
            )[:, -1]
            vocab_size = logits.shape[-1]
            if max(special) >= vocab_size:
                raise ValueError("Special token ID is outside the model vocabulary.")
            # Preserve the original model probabilities: mask AFTER log_softmax.
            log_probs = torch.log_softmax(logits.float(), dim=-1).reshape(batch_size, beams, vocab_size)
            if torch.isnan(log_probs).any():
                raise FloatingPointError("Nonfinite model probabilities during beam generation.")
            log_probs[:, :, model.pad_id] = -torch.inf
            log_probs[:, :, start_id] = -torch.inf

            # Store EOS candidates without removing other still-live alternatives.
            penalty = ((5 + length) / 6) ** length_penalty
            eos_scores = (live_scores + log_probs[:, :, end_id]) / penalty
            eos_tokens = torch.full((batch_size, beams, max_steps + 1), model.pad_id,
                                    dtype=torch.long, device=device)
            eos_tokens[:, :, :prefixes.shape[-1]] = prefixes
            eos_tokens[:, :, prefixes.shape[-1]] = end_id
            candidate_finished_scores = torch.cat([finished_scores, eos_scores], dim=1)
            candidate_finished_tokens = torch.cat([finished_tokens, eos_tokens], dim=1)
            candidate_finished_lengths = torch.cat([
                finished_lengths, torch.full((batch_size, beams), length, dtype=torch.long, device=device),
            ], dim=1)
            finished_scores, chosen = candidate_finished_scores.topk(beam_size, dim=1)
            finished_tokens = candidate_finished_tokens.gather(
                1, chosen.unsqueeze(-1).expand(-1, -1, max_steps + 1),
            )
            finished_lengths = candidate_finished_lengths.gather(1, chosen)

            if legal_tokens is None:
                vocabulary = torch.arange(vocab_size, device=device)
                legal_tokens = vocabulary[(vocabulary != model.pad_id) &
                                          (vocabulary != start_id) & (vocabulary != end_id)]
            if len(legal_tokens) == 0:
                live_scores.fill_(-torch.inf)
                break
            candidates = live_scores.unsqueeze(-1) + log_probs.index_select(-1, legal_tokens)
            next_beams = min(beam_size, beams * len(legal_tokens))
            live_scores, chosen = candidates.flatten(1).topk(next_beams, dim=1)
            parents = chosen // len(legal_tokens)
            next_tokens = legal_tokens[chosen % len(legal_tokens)]
            prefixes = torch.cat([
                prefixes.gather(1, parents.unsqueeze(-1).expand(-1, -1, prefixes.shape[-1])),
                next_tokens.unsqueeze(-1),
            ], dim=-1)
            live_length = length
            # Completed beams alone cannot justify early stopping under length normalization.
            if not torch.isfinite(live_scores).any():
                break

        live_penalty = ((5 + live_length) / 6) ** length_penalty
        final_scores = torch.cat([finished_scores, live_scores / live_penalty], dim=1)
        best_scores, selected = final_scores.max(dim=1)
        if not torch.isfinite(best_scores).all():
            raise FloatingPointError("No finite caption candidate survived beam generation.")
        selected = selected.tolist()
        finished_cpu = finished_tokens.cpu().tolist()
        lengths_cpu = finished_lengths.cpu().tolist()
        prefixes_cpu = prefixes.cpu().tolist()
        result = []
        for index, candidate in enumerate(selected):
            if candidate < beam_size:
                # Positions 1..L-1 are content; position L is EOS.
                length = lengths_cpu[index][candidate]
                result.append(finished_cpu[index][candidate][1:length])
            else:
                result.append(prefixes_cpu[index][candidate - beam_size][1:])
        return result
    finally:
        model.train(was_training)

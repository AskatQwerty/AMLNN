import hashlib
from pathlib import Path

import torch
from torch import nn


class GatedCrossAttentionLayer(nn.TransformerDecoderLayer):
    """A decoder block whose new visual residual can start at zero."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True, norm_first=True,
        )
        self.cross_gate = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, memory, causal_mask, text_padding_mask, memory_padding_mask):
        query = self.norm1(tokens)
        attended, _ = self.self_attn(
            query, query, query, attn_mask=causal_mask,
            key_padding_mask=text_padding_mask, need_weights=False,
        )
        tokens = tokens + self.dropout1(attended)

        # With gate=0, the copied image-prefix model is unchanged in eval mode.
        # The gate receives gradients immediately; the new branch learns as it opens.
        attended, _ = self.multihead_attn(
            self.norm2(tokens), memory, memory,
            key_padding_mask=memory_padding_mask, need_weights=False,
        )
        tokens = tokens + torch.tanh(self.cross_gate) * self.dropout2(attended)
        hidden = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tokens)))))
        return tokens + self.dropout3(hidden)


class FusionCaptionTransformer(nn.Module):
    """Frozen CNN vectors in, trainable visual fusion and caption decoder out."""

    def __init__(
        self, vocab_size, pad_id, grid_feature_dim=2048, region_feature_dim=1024,
        d_model=512, num_layers=3, nhead=8, dim_feedforward=2048,
        dropout=0.3, grid_size=7, max_regions=36, max_positions=34,
    ):
        super().__init__()
        if d_model % nhead or min(num_layers, grid_size, max_regions) < 1:
            raise ValueError("Invalid decoder width, layer count or visual layout.")
        if max_positions < 2:
            raise ValueError("Positions must accommodate the image prefix and a word.")
        self.decoder_type = "transformer_fusion"
        self.d_model = self.hidden_size = d_model
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.grid_size, self.max_regions = grid_size, max_regions
        self.max_positions = max_positions

        # Keep the trained global prefix. Averaging uses the cached ResNet50 grid.
        self.image_projection = nn.Linear(grid_feature_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_positions, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            GatedCrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size)

        # Fusion keeps all visual tokens; cross-attention learns how to use them.
        self.grid_projection = nn.Linear(grid_feature_dim, d_model)
        self.region_projection = nn.Linear(region_feature_dim, d_model)
        self.grid_norm = nn.LayerNorm(d_model)
        self.region_norm = nn.LayerNorm(d_model)
        self.row_embedding = nn.Embedding(grid_size, d_model)
        self.column_embedding = nn.Embedding(grid_size, d_model)
        self.source_embedding = nn.Embedding(2, d_model)
        self.region_box_projection = nn.Linear(4, d_model)

        for embedding in [self.embedding, self.position_embedding, self.row_embedding,
                          self.column_embedding, self.source_embedding]:
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()

    def prepare_memory(self, grid_features, region_features, region_mask, region_boxes=None, global_features=None):
        if grid_features.ndim != 3 or grid_features.shape[1:] != (
            self.grid_size ** 2, self.image_projection.in_features
        ):
            raise ValueError("Expected ResNet50 features [B,49,2048].")
        if region_features.ndim != 3 or len(region_features) != len(grid_features):
            raise ValueError("Expected region features [B,K,1024] for the same images.")
        if not 1 <= region_features.shape[1] <= self.max_regions:
            raise ValueError("The number of ROI slots is outside the configured range.")
        if region_features.shape[2] != self.region_projection.in_features:
            raise ValueError("Unexpected region feature dimension.")
        if region_mask.dtype != torch.bool or region_mask.shape != region_features.shape[:2]:
            raise ValueError("region_mask must be bool[B,K], True for valid regions.")
        region_mask = region_mask.to(grid_features.device)
        if not region_mask.any(dim=1).all():
            raise ValueError("Each image needs a valid ROI, including an extraction fallback.")
        if region_boxes is None:
            raise ValueError("Pass cached normalized xyxy boxes [B,K,4].")
        if region_boxes.shape != (*region_features.shape[:2], 4):
            raise ValueError("Expected normalized xyxy boxes [B,K,4].")

        # Ignore padded garbage before it reaches any trainable operation.
        clean_regions = region_features.masked_fill(~region_mask.unsqueeze(-1), 0)
        clean_boxes = region_boxes.masked_fill(~region_mask.unsqueeze(-1), 0)
        grid = self.grid_norm(self.grid_projection(grid_features))
        positions = torch.arange(self.grid_size ** 2, device=grid_features.device)
        grid_positions = self.row_embedding(positions // self.grid_size)
        grid_positions = grid_positions + self.column_embedding(positions % self.grid_size)
        grid = grid + grid_positions.unsqueeze(0) + self.source_embedding.weight[0]
        regions = self.region_norm(self.region_projection(clean_regions))
        regions = regions + self.region_box_projection(clean_boxes) + self.source_embedding.weight[1]
        regions = regions.masked_fill(~region_mask.unsqueeze(-1), 0)

        memory = torch.cat([grid, regions], dim=1)
        grid_is_padding = torch.zeros(grid.shape[:2], dtype=torch.bool, device=grid.device)
        memory_padding_mask = torch.cat([grid_is_padding, ~region_mask], dim=1)
        if global_features is None:
            global_features = grid_features.mean(dim=1)
        elif global_features.shape != (len(grid_features), self.image_projection.in_features):
            raise ValueError("Expected cached global ResNet50 vectors [B,2048].")
        image_token = self.image_projection(global_features).unsqueeze(1)
        return memory, memory_padding_mask, image_token

    def decode_memory(self, memory, memory_padding_mask, image_token, input_ids):
        if input_ids.ndim != 2 or len(input_ids) != len(memory) or input_ids.shape[1] < 1:
            raise ValueError("Each image needs a nonempty word sequence.")
        length = input_ids.shape[1] + 1
        if length > self.max_positions:
            raise ValueError("The image prefix and input words exceed the learned positions.")
        words = self.embedding(input_ids) * self.d_model ** 0.5
        tokens = torch.cat([image_token, words], dim=1)
        positions = torch.arange(length, device=input_ids.device)
        tokens = self.embedding_dropout(tokens + self.position_embedding(positions).unsqueeze(0))
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), 1)
        image_is_padding = torch.zeros(len(input_ids), 1, dtype=torch.bool, device=tokens.device)
        text_padding = torch.cat([image_is_padding, input_ids.eq(self.pad_id)], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, memory, causal, text_padding, memory_padding_mask)
        return self.output(self.output_dropout(self.final_norm(tokens[:, 1:])))

    def forward(self, grid_features, region_features, input_ids, region_mask, region_boxes=None, global_features=None):
        memory, padding, image_token = self.prepare_memory(
            grid_features, region_features, region_mask, region_boxes, global_features
        )
        return self.decode_memory(memory, padding, image_token, input_ids)

    @torch.no_grad()
    def greedy(
        self, grid_features, region_features, start_id, end_id, max_steps=33,
        region_mask=None, region_boxes=None, global_features=None,
    ):
        if not 1 <= max_steps < self.max_positions:
            raise ValueError("Generation must fit the image prefix and learned text positions.")
        was_training = self.training
        self.eval()
        try:
            memory, padding, image_token = self.prepare_memory(
                grid_features, region_features, region_mask, region_boxes, global_features
            )
            prefix = torch.full((len(grid_features), 1), start_id, dtype=torch.long,
                                device=grid_features.device)
            finished = torch.zeros(len(prefix), dtype=torch.bool, device=prefix.device)
            predictions = [[] for _ in range(len(prefix))]
            for _ in range(max_steps):
                logits = self.decode_memory(memory, padding, image_token, prefix)[:, -1]
                logits[:, self.pad_id] = -torch.inf
                logits[:, start_id] = -torch.inf
                next_ids = logits.argmax(dim=-1)
                for index, (token_id, done) in enumerate(zip(next_ids.tolist(), finished.tolist())):
                    if not done and token_id != end_id:
                        predictions[index].append(token_id)
                finished |= next_ids.eq(end_id)
                if finished.all():
                    break
                prefix = torch.cat([prefix, next_ids.unsqueeze(1)], dim=1)
            return predictions
        finally:
            self.train(was_training)

    def cross_gate_values(self):
        return [float(torch.tanh(layer.cross_gate.detach()).cpu()) for layer in self.layers]


def warm_start_from_prefix(model, checkpoint_path, expected_sha256=None):
    """Copy the existing trained caption core; leave the new cross-attention branch new."""
    checkpoint_path = Path(checkpoint_path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("The chosen pretrained caption checkpoint changed.")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    target = model.state_dict()
    copied, consumed = {}, set()
    for name in target:
        origin = None
        if name.startswith(("image_projection.", "embedding.", "position_embedding.",
                            "final_norm.", "output.")):
            origin = name
        elif name.startswith("grid_projection."):
            # Initialize the new grid projection from the useful existing visual mapping.
            origin = name.replace("grid_projection.", "image_projection.", 1)
        elif name.startswith("layers."):
            module = name.split(".")[2]
            if module in {"self_attn", "linear1", "linear2", "norm1"}:
                origin = name
            elif module == "norm3":
                origin = name.replace(".norm3.", ".norm2.")
        if origin is not None:
            if origin not in source or source[origin].shape != target[name].shape:
                raise ValueError(f"Incompatible caption warm-start parameter: {name} <- {origin}")
            target[name] = source[origin].detach().clone()
            copied[name] = origin
            consumed.add(origin)
    if set(source) != consumed:
        raise ValueError(f"Unexpected source checkpoint keys: {sorted(set(source) - consumed)}")
    model.load_state_dict(target, strict=True)
    with torch.no_grad():
        for layer in model.layers:
            layer.cross_gate.zero_()
    all_parameters = dict(model.named_parameters())
    return {
        "source_checkpoint": str(checkpoint_path.resolve()), "source_sha256": digest,
        "copied_parameters": copied,
        "new_parameters": [name for name in all_parameters if name not in copied],
        "copied_parameter_count": sum(all_parameters[name].numel() for name in copied),
        "new_parameter_count": sum(parameter.numel() for name, parameter in all_parameters.items() if name not in copied),
        "total_trainable_parameters": sum(parameter.numel() for parameter in all_parameters.values()),
        "initial_cross_gates": model.cross_gate_values(),
        "initial_behavior": "Copied image-prefix model in eval mode on the supplied global features (mean(grid) fallback); cross-attention residuals initially zero. This is initialization, not a new achieved validation result.",
        "source_unchanged": True,
    }


def fusion_parameter_groups(model, core_lr=1e-4, fusion_lr=5e-4):
    """Use a smaller learning rate for the transferred caption core."""
    core, fusion = [], []
    core_names, fusion_names = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_core = name.startswith(("image_projection.", "embedding.", "position_embedding.",
                                   "final_norm.", "output."))
        if name.startswith("layers."):
            is_core = name.split(".")[2] in {"self_attn", "linear1", "linear2", "norm1", "norm3"}
        if is_core:
            core.append(parameter)
            core_names.append(name)
        else:
            fusion.append(parameter)
            fusion_names.append(name)
    assert len({id(parameter) for parameter in core + fusion}) == len(core) + len(fusion)
    assert len(core) + len(fusion) == sum(parameter.requires_grad for parameter in model.parameters())
    groups = [
        {"params": core, "lr": core_lr, "group_name": "transferred_caption_core"},
        {"params": fusion, "lr": fusion_lr, "group_name": "new_fusion_cross_attention"},
    ]
    report = {"core_lr": core_lr, "fusion_lr": fusion_lr,
              "core_parameter_names": core_names, "fusion_parameter_names": fusion_names,
              "core_parameters": sum(p.numel() for p in core), "fusion_parameters": sum(p.numel() for p in fusion)}
    return groups, report

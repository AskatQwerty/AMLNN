import torch
from torch import nn


class CaptionRegionLSTM(nn.Module):
    """The spatial LSTM decoder with a variable number of valid visual regions."""

    def __init__(
        self,
        feature_dim,
        vocab_size,
        pad_id,
        embedding_size=512,
        hidden_size=512,
        num_layers=3,
        dropout=0.3,
    ):
        super().__init__()
        if embedding_size != hidden_size:
            raise ValueError("This model uses equal embedding and hidden sizes.")
        if num_layers < 1:
            raise ValueError("Use at least one recurrent layer.")

        self.decoder_type = "lstm_attention"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_id = pad_id

        # Keep the original modules and construction order for dense-grid comparisons.
        self.image_projection = nn.Linear(feature_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attention_visual = nn.Linear(hidden_size, hidden_size)
        self.attention_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.input_fusion = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.recurrent = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def prepare_image(self, features, visual_mask=None):
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("Expected visual features with shape [batch, K, feature_dim], K >= 1.")
        if features.shape[2] != self.image_projection.in_features:
            raise ValueError("The cached feature dimension does not match the model.")
        if visual_mask is not None:
            if not isinstance(visual_mask, torch.Tensor) or visual_mask.dtype != torch.bool:
                raise TypeError("visual_mask must be a boolean tensor; True means valid.")
            if visual_mask.shape != features.shape[:2]:
                raise ValueError("visual_mask must have shape [batch, K].")
            visual_mask = visual_mask.to(features.device)
            if not visual_mask.any(dim=1).all():
                raise ValueError("Each image must have at least one valid visual region.")
            if visual_mask.all():
                # Preserve the original dense-grid calculation exactly.
                visual_mask = None

        if visual_mask is not None:
            # Remove padded garbage, including NaNs, before any learned operation.
            features = features.masked_fill(~visual_mask.unsqueeze(-1), 0)
        visual = self.image_projection(features)
        if visual_mask is not None:
            visual = visual.masked_fill(~visual_mask.unsqueeze(-1), 0)
        return visual, visual_mask

    def initial_state(self, visual, visual_mask=None):
        hidden = visual.new_zeros(self.num_layers, len(visual), self.hidden_size)
        initial = (hidden, torch.zeros_like(hidden))
        if visual_mask is None:
            image_mean = visual.mean(dim=1)
        else:
            valid_values = visual.masked_fill(~visual_mask.unsqueeze(-1), 0)
            valid_count = visual_mask.sum(dim=1, keepdim=True).to(visual.dtype)
            image_mean = valid_values.sum(dim=1) / valid_count
        # Process the mean image separately; this step has no word target.
        _, state = self.recurrent(image_mean.unsqueeze(1), initial)
        return state

    def image_state(self, features, visual_mask=None):
        visual, visual_mask = self.prepare_image(features, visual_mask)
        return self.initial_state(visual, visual_mask)

    def word_step(self, token_ids, visual, visual_keys, state, visual_mask=None):
        query = self.attention_hidden(state[0][-1]).unsqueeze(1)
        scores = self.attention_score(torch.tanh(visual_keys + query)).squeeze(-1)
        if visual_mask is not None:
            scores = scores.masked_fill(~visual_mask, -torch.inf)
        attention = torch.softmax(scores, dim=1)
        context = torch.sum(visual * attention.unsqueeze(-1), dim=1)

        word = self.embedding_dropout(self.embedding(token_ids))
        recurrent_input = self.input_fusion(torch.cat([word, context], dim=-1))
        output, state = self.recurrent(recurrent_input.unsqueeze(1), state)
        logits = self.output(self.output_dropout(output[:, 0]))
        return logits, state, attention

    def forward_with_attention(self, features, input_ids, visual_mask=None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or len(features) != len(input_ids):
            raise ValueError("Each image needs a nonempty input token sequence.")
        visual, visual_mask = self.prepare_image(features, visual_mask)
        visual_keys = self.attention_visual(visual)
        state = self.initial_state(visual, visual_mask)
        logits = []
        attention = []

        # Input: <start> and previous words; outputs predict the next words.
        for token_ids in input_ids.unbind(dim=1):
            next_logits, state, weights = self.word_step(
                token_ids, visual, visual_keys, state, visual_mask
            )
            logits.append(next_logits)
            attention.append(weights)
        return torch.stack(logits, dim=1), torch.stack(attention, dim=1)

    def forward(self, features, input_ids, visual_mask=None):
        logits, _ = self.forward_with_attention(features, input_ids, visual_mask)
        return logits

    @torch.no_grad()
    def greedy(self, features, start_id, end_id, max_steps=33, visual_mask=None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        was_training = self.training
        self.eval()
        try:
            visual, visual_mask = self.prepare_image(features, visual_mask)
            visual_keys = self.attention_visual(visual)
            state = self.initial_state(visual, visual_mask)
            current = torch.full(
                (len(features),), start_id, dtype=torch.long, device=features.device
            )
            finished = torch.zeros(len(features), dtype=torch.bool, device=features.device)
            predictions = [[] for _ in range(len(features))]
            for _ in range(max_steps):
                logits, state, _ = self.word_step(
                    current, visual, visual_keys, state, visual_mask
                )
                logits[:, self.pad_id] = -torch.inf
                logits[:, start_id] = -torch.inf
                next_ids = logits.argmax(dim=-1)
                for index, (token_id, done) in enumerate(zip(next_ids.tolist(), finished.tolist())):
                    if not done and token_id != end_id:
                        predictions[index].append(token_id)
                finished |= next_ids.eq(end_id)
                if finished.all():
                    break
                current = next_ids
            return predictions
        finally:
            self.train(was_training)

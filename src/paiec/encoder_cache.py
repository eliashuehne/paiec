"""Bounded, process-local FP32 features for a frozen sequential encoder."""
from collections import OrderedDict
import hashlib
from types import SimpleNamespace


class FrozenEncoderCache:
    def __init__(self, encoder, max_bytes, *, identity):
        if max_bytes <= 0:
            raise ValueError("Cache byte limit must be positive")
        self.encoder = encoder
        self.original = encoder.forward
        self.max_bytes = int(max_bytes)
        self.identity = identity
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.versions = self._versions()
        self.closed = False
        self._check()
        encoder.forward = self.forward

    def _versions(self):
        return tuple((id(p), p._version) for p in self.encoder.parameters())

    def _check(self):
        import torch
        if self.encoder.training or any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError("Encoder cache requires frozen parameters and evaluation mode")
        if any(p.dtype != torch.float32 for p in self.encoder.parameters()):
            raise RuntimeError("Encoder cache requires FP32 weights")
        if self._versions() != self.versions:
            raise RuntimeError("Frozen encoder changed; discard its feature cache")

    def forward(self, input_ids, attention_mask, **kwargs):
        import torch
        if kwargs or input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Cache supports only sequential token IDs and a 2D right-padding mask")
        self._check()
        ids = input_ids.detach().cpu()
        masks = attention_mask.detach().cpu()
        lengths = masks.sum(1).long().tolist()
        keys = []
        for row, mask, length in zip(ids, masks, lengths):
            if length < 1 or not bool((mask[:length] == 1).all()) or bool(mask[length:].any()):
                raise ValueError("Encoder cache requires nonempty, right-padded inputs")
            keys.append(hashlib.sha256(row[:length].contiguous().numpy().tobytes()).digest())
        h = torch.zeros((*input_ids.shape, self.encoder.config.hidden_size),
                        dtype=torch.float32, device=input_ids.device)
        missing = []
        for i, key in enumerate(keys):
            value = self.entries.get(key)
            if value is None:
                self.misses += 1
                missing.append(i)
            else:
                self.hits += 1
                self.entries.move_to_end(key)
                h[i, :lengths[i]] = value
        if missing:
            with torch.no_grad():
                hidden = self.original(input_ids=input_ids[missing],
                                       attention_mask=attention_mask[missing]).last_hidden_state
            for j, i in enumerate(missing):
                value = hidden[j, :lengths[i]].detach().clone()
                h[i, :lengths[i]] = value
                size = value.numel() * value.element_size()
                if size <= self.max_bytes:
                    old = self.entries.pop(keys[i], None)
                    if old is not None:
                        self.bytes -= old.numel() * old.element_size()
                    while self.bytes + size > self.max_bytes:
                        _, old = self.entries.popitem(last=False)
                        self.bytes -= old.numel() * old.element_size()
                        self.evictions += 1
                    self.entries[keys[i]] = value
                    self.bytes += size
        return SimpleNamespace(last_hidden_state=h)

    def stats(self):
        return dict(identity=self.identity, max_bytes=self.max_bytes, bytes=self.bytes,
                    entries=len(self.entries), hits=self.hits, misses=self.misses,
                    evictions=self.evictions, dtype="float32")

    def close(self):
        if not self.closed:
            self.encoder.forward = self.original
            self.entries.clear()
            self.bytes = 0
            self.closed = True


def validate_cache(model, cache, batches, device):
    """Compare native/cached probabilities and gradients with identical dropout RNG."""
    import torch
    from laya.train import _forward
    from .head_training import freeze_for_head
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    probability_drift = gradient_drift = 0.
    try:
        for number, batch in enumerate(batches):
            model.eval()
            with torch.no_grad():
                model.encoder.forward = cache.original
                old = _forward(model, batch, device, False, True).softmax(-1)
                model.encoder.forward = cache.forward
                new = _forward(model, batch, device, False, True).softmax(-1)
                probability_drift = max(probability_drift, float((old-new).abs().max()))
            if number < 4:
                freeze_for_head(model)
                rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
                model.encoder.forward = cache.original
                _forward(model, batch, device, False, True).square().mean().backward()
                gradients = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.set_rng_state(rng, device)
                else:
                    torch.set_rng_state(rng)
                model.encoder.forward = cache.forward
                _forward(model, batch, device, False, True).square().mean().backward()
                gradient_drift = max(gradient_drift, max(float((p.grad-gradients[n]).abs().max())
                    for n,p in model.named_parameters() if n in gradients))
                model.zero_grad(set_to_none=True)
        if probability_drift > 1e-5 or gradient_drift > 1e-4:
            raise RuntimeError(f"Cache equivalence failed: probability={probability_drift}, gradient={gradient_drift}")
        return dict(max_probability_difference=probability_drift,
                    max_gradient_difference=gradient_drift, n_batches=len(batches))
    finally:
        model.encoder.forward = cache.forward
        model.zero_grad(set_to_none=True)
        freeze_for_head(model)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)

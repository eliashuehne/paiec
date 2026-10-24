"""One native view scorer and aggregation rule shared by trainer and predictor."""
import math


def probability_from_logits(logits, weights):
    if len(logits) != len(weights) or not logits or any(not math.isfinite(w) or w <= 0 for w in weights) or abs(sum(weights)-1.) > 1e-6:
        raise ValueError("Expected positive normalized view weights")
    mean = [sum(w * float(z[k]) for w, z in zip(weights, logits)) for k in [0, 1]]
    if not all(math.isfinite(v) for v in mean):
        raise ValueError("Non-finite grouped logits")
    delta = mean[1] - mean[0]
    return 1. / (1. + math.exp(-delta)) if delta >= 0 else math.exp(delta) / (1. + math.exp(delta))


class TorchGroupedEngine:
    def __init__(self, model, tokenizer, *, device="cpu", precision="float32", max_len=8192, upper_layers=2):
        import torch
        if precision not in {"float16", "float32"} or (precision == "float16" and torch.device(device).type != "cuda"):
            raise ValueError("Grouped FP16 requires the qualified CUDA path; CPU/MPS use FP32")
        self.model, self.tokenizer = model, tokenizer
        self.device, self.precision, self.max_len, self.upper_layers = torch.device(device), precision, max_len, upper_layers
        self.model.to(self.device)

    def view_logits(self, view):
        from .encoder_training import collate, forward
        if len(view["ids"]) > self.max_len:
            raise ValueError("Grouped view exceeds the checkpoint context limit")
        batch = collate([view], self.tokenizer.pad_token_id, self.max_len)
        return forward(self.model, batch, self.device, self.precision, self.upper_layers, fp32_scorer=True).reshape(2)

    def predict_group(self, group):
        import torch
        self.model.eval()
        with torch.no_grad():
            logits = [self.view_logits(view).float().cpu().tolist() for view in group["views"]]
        return float(probability_from_logits(logits, group["weights"]))

    def close(self):
        self.model = None

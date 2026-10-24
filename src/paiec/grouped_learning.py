"""One loss per original example, with bounded graph memory across views."""


def backward_group(forward_view, views, weights, label, *, divisor=1., scaler=None, cuda=False):
    import torch
    from torch.nn.functional import cross_entropy
    if len(views) != len(weights) or not views or abs(sum(weights) - 1.) > 1e-6 or any(w <= 0 for w in weights):
        raise ValueError("Each original example needs normalized positive view weights")
    target = torch.tensor([int(label)], device="cuda" if cuda else "cpu")
    if len(views) == 1:
        logits = forward_view(views[0]).reshape(1, 2)
        loss = cross_entropy(logits, target) / divisor
        (scaler.scale(loss) if scaler else loss).backward()
        return float(loss.detach())
    # Capture dropout RNG states during a graph-free first pass, then replay each
    # view for its exact chain-rule contribution. Only one view graph stays live.
    states, logits = [], []
    with torch.no_grad():
        for view in views:
            states.append((torch.get_rng_state(), torch.cuda.get_rng_state() if cuda else None))
            logits.append(forward_view(view).reshape(2).detach())
        mean = sum(w * value for w, value in zip(weights, logits))
        derivative = mean.softmax(-1)
        derivative[int(label)] -= 1.
        derivative /= divisor
        value = cross_entropy(mean.reshape(1, 2), target) / divisor
    after = torch.get_rng_state(), torch.cuda.get_rng_state() if cuda else None
    try:
        for view, weight, (cpu_state, cuda_state) in zip(views, weights, states):
            torch.set_rng_state(cpu_state)
            if cuda:
                torch.cuda.set_rng_state(cuda_state)
            contribution = (forward_view(view).reshape(2) * derivative * weight).sum()
            (scaler.scale(contribution) if scaler else contribution).backward()
    finally:
        torch.set_rng_state(after[0])
        if cuda:
            torch.cuda.set_rng_state(after[1])
    return float(value)

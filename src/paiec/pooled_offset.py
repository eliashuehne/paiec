"""The prior offset with each benchmark's shared parts learned from every system.

The model is in ``pooling``. ``fit`` first fits the prior offset exactly as
``PriorOffset`` does (shift, scale, tau and sigma from each system's own
labels), then chooses ``share`` and ``item_sigma`` on the training benchmarks:
each benchmark's sampled pairs are cut into pools of ``FIT_POOL_SIZE`` and
every pair is scored with its pool's labels, using the same out-of-fold prior
logits. ``share = 0`` and ``item_sigma = 0`` still pool the tag shifts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .groups import TagDesign, parse_tags
from .metrics import BUDGETS, alc_weights
from .pooling import Layout, PoolPosterior, PoolPrior, index_of, item_shifts
from .prior_offset import PriorOffset
from .split import stable_seed
from .update import laplace_posterior, sigmoid


def subject_key(subject: dict) -> tuple:
    """Visible fields as text, so missing values compare equal across copies."""
    return tuple(sorted((str(key), str(value)) for key, value in subject.items()))


def item_key(item: dict) -> tuple:
    return item.get("item_content"), item.get("item_features")


def make_member_pools(held_out: dict, size: int, seed: int = 2) -> list[list]:
    """Cut each benchmark's (episode, revealed logits, evaluation logits) into seeded pools."""
    pools = []
    for benchmark in sorted(held_out):
        members = held_out[benchmark]
        order = np.random.default_rng(stable_seed(seed, benchmark, "pools")).permutation(len(members))
        members = [members[index] for index in order]
        pools.extend(members[start : start + size] for start in range(0, len(members), size))
    return pools


@dataclass
class MemberEval:
    """One pool member's evaluation items, arranged against the pool's tag columns."""

    starts: np.ndarray
    tags: np.ndarray
    unseen: np.ndarray
    items: np.ndarray
    rows: np.ndarray
    correct: np.ndarray
    weight: float


@dataclass
class PoolState:
    """One pool at one budget: everything that does not depend on ``share`` or ``item_sigma``."""

    starts: np.ndarray
    subject_index: np.ndarray
    tag_design: TagDesign
    labels: np.ndarray
    item_index: np.ndarray
    n_items: int
    members: list[MemberEval]


def pool_state(pool: list, budget: int, start: Callable) -> PoolState:
    """Shared revealed labels of a pool at one budget, and each member's evaluation items."""
    starts, subjects, tags, items, labels = [], [], [], [], []
    for number, (episode, revealed, _) in enumerate(pool):
        starts.append(start(revealed[:budget]))
        subjects += [number] * budget
        tags += episode["revealed_tags"][:budget]
        items += episode["revealed_ids"][:budget]
        labels.append(np.asarray(episode["revealed_labels"][:budget], dtype=float))
    design = TagDesign.from_revealed(tags)
    item_column, item_index = index_of(items)
    members = []
    for episode, _, evaluated in pool:
        eval_tags = np.zeros((len(episode["eval_tags"]), design.matrix.shape[1] - 1))
        unseen = np.zeros(len(episode["eval_tags"]))
        for row, target_tags in enumerate(episode["eval_tags"]):
            seen, unseen[row] = design.target_columns(target_tags)
            eval_tags[row, [column - 1 for column in seen]] = 1.0
        members.append(MemberEval(
            start(evaluated), eval_tags, unseen,
            np.asarray([item_column.get(item, -1) for item in episode["eval_ids"]], dtype=int),
            np.asarray(episode["eval_rows"], dtype=float),
            np.asarray(episode["eval_correct"], dtype=float), float(episode["weight"]),
        ))
    _, subject_index = index_of(subjects)
    return PoolState(np.concatenate(starts), subject_index, design, np.concatenate(labels),
                     item_index, len(item_column), members)


def pooled_scores(pools: list[list[PoolState]], prior: PoolPrior, item_sigmas) -> np.ndarray:
    """Weighted mean ALC of every member, one value per ``item_sigma``.

    Same arithmetic as ``PoolPosterior.predict``, vectorised per member.
    """
    totals, total_weight = np.zeros(len(item_sigmas)), 0.0
    for states in pools:
        member_alc = np.zeros((len(states[0].members), len(item_sigmas)))
        for alc_weight, state in zip(alc_weights(BUDGETS), states):
            n_subjects = len(np.unique(state.subject_index))
            n_tags = state.tag_design.matrix.shape[1] - 1 if prior.sigma > 0 else 0
            layout = Layout.build(n_subjects, n_tags, prior)
            design = layout.design(state.subject_index, state.tag_design.matrix)
            mean, covariance = laplace_posterior(state.starts, design, state.labels, layout.precision)
            fitted = state.starts + design @ mean
            item_fits = [item_shifts(fitted, state.item_index, state.labels, state.n_items, s)
                         for s in item_sigmas]
            for number, member in enumerate(state.members):
                z = np.zeros((len(member.starts), len(layout.precision)))
                if layout.level is not None:
                    z[:, layout.level] = 1.0
                missing = 0.0
                if number < n_subjects:
                    z[:, layout.own + number] = 1.0
                else:
                    missing = prior.own_variance
                if n_tags:
                    z[:, layout.tags :] = member.tags
                    missing = missing + member.unseen * prior.sigma * prior.sigma
                base_mean = member.starts + z @ mean
                base_variance = np.einsum("ij,jk,ik->i", z, covariance, z) + missing
                revealed = member.items >= 0
                for column, (item_mean, item_variance) in enumerate(item_fits):
                    logit, variance = base_mean.copy(), base_variance.copy()
                    logit[revealed] += item_mean[member.items[revealed]]
                    variance[revealed] += item_variance[member.items[revealed]]
                    p = sigmoid(logit / np.sqrt(1.0 + np.pi * variance / 8.0))
                    squared = np.sum(member.rows * p * p - 2 * p * member.correct + member.correct)
                    member_alc[number, column] += alc_weight * squared / np.sum(member.rows)
        weights = np.asarray([member.weight for member in states[0].members])
        totals += weights @ member_alc
        total_weight += weights.sum()
    return totals / total_weight


@dataclass
class PooledOffset(PriorOffset):
    """Prior offset whose benchmark level, tag shifts and question shifts pool all systems."""

    FIT_POOL_SIZE = 5
    SHARES = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8)
    ITEM_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

    share: float = 0.0
    item_sigma: float = 0.0
    fit_scores: np.ndarray | None = field(default=None, repr=False, compare=False)
    _labeled: object = field(default=None, init=False, repr=False, compare=False)
    _labeled_size: int = field(default=-1, init=False, repr=False, compare=False)
    _pools: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def fit(self, data_root, train_benchmarks):
        base, held_out = self.held_out_episodes(data_root, train_benchmarks)
        self.fit_held_out(base, held_out)
        self.fit_pooling(held_out)
        self.prior = self.PRIOR_CLASS().fit(data_root, train_benchmarks)
        return self

    def fit_pooling(self, held_out: dict):
        """Grid ``share`` x ``item_sigma`` with the prior offset's other parameters fixed."""
        def start(logits):
            return self.level + self.shift + self.scale * (np.asarray(logits, dtype=float) - self.center)

        pools = [[pool_state(pool, budget, start) for budget in BUDGETS]
                 for pool in make_member_pools(held_out, self.FIT_POOL_SIZE)]
        self.fit_scores = np.array([
            pooled_scores(pools, PoolPrior(self.tau, share, self.sigma, 0.0), self.ITEM_SIGMAS)
            for share in self.SHARES
        ])
        row, column = np.unravel_index(np.argmin(self.fit_scores), self.fit_scores.shape)
        self.share, self.item_sigma = float(self.SHARES[row]), float(self.ITEM_SIGMAS[column])
        self._posteriors.clear()
        return self

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        posterior = self._pool(item.get("benchmark_id"), labeled)
        return posterior.predict(self._logit(subject, item), subject_key(subject),
                                 parse_tags(item.get("item_features", "")), item_key(item))

    def _pool(self, benchmark, labeled) -> PoolPosterior:
        # Targets of one checkpoint share the same ``labeled`` list; fit each benchmark once.
        size = len(labeled or ())
        if labeled is not self._labeled or size != self._labeled_size:
            self._labeled, self._labeled_size, self._pools = labeled, size, {}
        if benchmark not in self._pools:
            examples = [(s, e, int(y)) for (s, e), y in labeled or () if e.get("benchmark_id") == benchmark]
            key = (benchmark, tuple((subject_key(s), item_key(e), y) for s, e, y in examples))
            if key not in self._posteriors:
                if len(self._posteriors) >= self.CACHE_LIMIT:
                    self._posteriors.clear()
                self._posteriors[key] = PoolPosterior.fit(
                    [self._logit(s, e) for s, e, _ in examples],
                    [subject_key(s) for s, _, _ in examples],
                    [parse_tags(e.get("item_features", "")) for _, e, _ in examples],
                    [item_key(e) for _, e, _ in examples],
                    [y for _, _, y in examples],
                    PoolPrior(self.tau, self.share, self.sigma, self.item_sigma),
                )
            self._pools[benchmark] = self._posteriors[key]
        return self._pools[benchmark]


__all__ = ["PooledOffset", "make_member_pools", "pool_state", "pooled_scores"]

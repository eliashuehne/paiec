"""End to end: a per-item prior feeding the tag offset.

``logit = level + shift + scale * (prior_logit(subject, item) - center) + pair_shift + tag_shifts``

The prior is any model with ``fit(data_root, benchmarks)`` and ``logit(subject, item)``
(currently ``LogisticPrior``). ``scale = 0`` is exactly the tag offset, so the fit can
only move away from it if the prior helps on held-out data. To avoid trusting a prior
that memorised its own training benchmarks, the prior logits used for fitting
``shift``, ``scale``, ``tau`` and ``sigma`` are out-of-fold: each training benchmark is
scored by a prior fitted without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .episodes import tag_episodes
from .features import DIM, FEATURE_VERSION, item_vector, subject_vector
from .folds import benchmark_folds
from .gbdt_prior import GBDTPrior
from .groups import TagDesign, parse_tags
from .metrics import BUDGETS, alc_weights
from .offset import BayesianOffset, TagPosterior
from .prior import LogisticPrior
from .schema import item_record, load_benchmark, subject_record
from .shrinkage import matching_examples
from .update import sigmoid, tag_offset_posterior


def _vector_cache(data_root, benchmark: str) -> Path:
    name = benchmark.split("/")[0]
    return Path(data_root).resolve().parent / "cache" / f"episode_vectors_v{FEATURE_VERSION}" / f"{name}.npz"


def _pack(vectors: list) -> dict:
    lengths = [len(indices) for indices, _ in vectors]
    return {
        "indptr": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "indices": np.concatenate([i for i, _ in vectors]) if vectors else np.zeros(0, np.int64),
        "values": np.concatenate([v for _, v in vectors]) if vectors else np.zeros(0),
    }


def _unpack(indptr, indices, values) -> list:
    return [(indices[a:b], values[a:b]) for a, b in zip(indptr[:-1], indptr[1:])]


def episode_vectors(data_root, benchmark: str) -> tuple[dict, dict]:
    """Feature vectors for every subject and (subject, item) in the benchmark's tag episodes.

    Built once from the item text (the item's interactors come from that
    subject's first response, as in the harness) and cached on disk.
    """
    path = _vector_cache(data_root, benchmark)
    if path.is_file():
        with np.load(path, allow_pickle=False) as stored:
            subjects = dict(zip(stored["subject_ids"], _unpack(stored["s_indptr"], stored["s_indices"], stored["s_values"])))
            rows = _unpack(stored["i_indptr"], stored["i_indices"], stored["i_values"])
            items = {(s, i): rows[r] for s, i, r in zip(stored["pair_subjects"], stored["pair_items"], stored["pair_rows"])}
        return subjects, items

    name = benchmark.split("/")[0]
    episodes = tag_episodes(str(data_root), benchmark)
    tables = load_benchmark(data_root, name)
    interactors = tables.responses.drop_duplicates(["subject_id", "item_id"]).set_index(["subject_id", "item_id"])["interactors"]
    subject_ids = sorted({e["subject_id"] for e in episodes})
    subject_vectors = [subject_vector(subject_record(tables.subjects.loc[s])) for s in subject_ids]

    row_of, item_vectors, pair_subjects, pair_items, pair_rows = {}, [], [], [], []
    for episode in episodes:
        s = episode["subject_id"]
        for item_id in dict.fromkeys(episode["revealed_ids"] + episode["eval_ids"]):
            stored = tables.items.loc[item_id]
            record = {"content": stored["content"], "item_features": stored["item_features"],
                      "interactors": interactors.get((s, item_id), "")}
            key = (item_id, str(record["interactors"]))
            if key not in row_of:
                row_of[key] = len(item_vectors)
                item_vectors.append(item_vector(item_record(record, name)))
            pair_subjects.append(s)
            pair_items.append(item_id)
            pair_rows.append(row_of[key])

    s_pack, i_pack = _pack(subject_vectors), _pack(item_vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        subject_ids=np.asarray(subject_ids, dtype=str), s_indptr=s_pack["indptr"],
        s_indices=s_pack["indices"], s_values=s_pack["values"],
        i_indptr=i_pack["indptr"], i_indices=i_pack["indices"], i_values=i_pack["values"],
        pair_subjects=np.asarray(pair_subjects, dtype=str), pair_items=np.asarray(pair_items, dtype=str),
        pair_rows=np.asarray(pair_rows, dtype=np.int64),
    )
    return episode_vectors(data_root, benchmark)


def episode_logits(prior, vectors: tuple[dict, dict], episodes: Sequence[dict]) -> list:
    """(revealed prior logits, evaluation prior logits) for each episode, scored in one batch."""
    subjects, items = vectors
    indptr, indices, values, sizes = [0], [], [], []
    for episode in episodes:
        s = episode["subject_id"]
        for ids in (episode["revealed_ids"], episode["eval_ids"]):
            sizes.append(len(ids))
            for item_id in ids:
                item_vec = items[(s, item_id)]
                indices.append(np.concatenate([item_vec[0], subjects[s][0]]))
                values.append(np.concatenate([item_vec[1], subjects[s][1]]))
                indptr.append(indptr[-1] + len(indices[-1]))
    if not indices:
        return [(np.zeros(0), np.zeros(0)) for _ in episodes]
    rows = csr_matrix((np.concatenate(values), np.concatenate(indices), indptr), shape=(len(indptr) - 1, DIM))
    logits = np.split(prior.row_logits(rows), np.cumsum(sizes)[:-1])
    return list(zip(logits[0::2], logits[1::2]))


@dataclass
class PriorState:
    """One episode at one budget, with everything that does not depend on the parameters."""

    design: np.ndarray
    labels: np.ndarray
    revealed_logits: np.ndarray
    eval_design: np.ndarray
    eval_unseen: np.ndarray
    eval_logits: np.ndarray
    eval_rows: np.ndarray
    eval_correct: np.ndarray


def prior_states(episode: dict, revealed_logits, eval_logits) -> list[PriorState]:
    states = []
    for budget in BUDGETS:
        design = TagDesign.from_revealed(episode["revealed_tags"][:budget])
        eval_design = np.zeros((len(episode["eval_tags"]), design.matrix.shape[1]))
        eval_design[:, 0] = 1.0
        unseen = np.zeros(len(episode["eval_tags"]))
        for row, tags in enumerate(episode["eval_tags"]):
            columns, unseen[row] = design.target_columns(tags)
            eval_design[row, columns] = 1.0
        states.append(PriorState(
            design.matrix, np.asarray(episode["revealed_labels"][:budget], dtype=float),
            np.asarray(revealed_logits[:budget], dtype=float), eval_design, unseen,
            np.asarray(eval_logits, dtype=float), np.asarray(episode["eval_rows"], dtype=float),
            np.asarray(episode["eval_correct"], dtype=float),
        ))
    return states


def prior_offset_alc(states: list, weights, level: float, shift: float, scale: float, center: float,
                     tau: float, sigma: float) -> float:
    """Weighted mean ALC over episodes; same arithmetic as ``PriorOffset.__call__``, vectorised per episode."""
    scores = []
    for pair_states in states:
        alc = 0.0
        for alc_weight, state in zip(alc_weights(BUDGETS), pair_states):
            revealed = level + shift + scale * (state.revealed_logits - center)
            target = level + shift + scale * (state.eval_logits - center)
            mean, covariance = tag_offset_posterior(revealed, state.design, state.labels, tau, sigma)
            z = state.eval_design[:, : len(mean)]
            variance = np.einsum("ij,jk,ik->i", z, covariance, z)
            if sigma > 0:
                variance = variance + state.eval_unseen * sigma * sigma
            p = sigmoid((target + z @ mean) / np.sqrt(1.0 + np.pi * variance / 8.0))
            squared = np.sum(state.eval_rows * p * p - 2 * p * state.eval_correct + state.eval_correct)
            alc += alc_weight * squared / np.sum(state.eval_rows)
        scores.append(alc)
    return float(np.average(scores, weights=weights))


@dataclass
class PriorOffset:
    """Per-item prior logits (scaled and shifted) under the Bayesian tag offset."""

    PRIOR_CLASS = LogisticPrior
    INNER_FOLDS = 3
    SHIFTS = (-0.2, 0.0, 0.2)
    SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    TAU_SCALES = (0.7, 0.85, 1.0)
    SIGMAS = (0.5, 0.75, 1.0)
    CACHE_LIMIT = 4096

    level: float = 0.0
    shift: float = 0.0
    scale: float = 0.0
    center: float = 0.0
    tau: float = 1.0
    sigma: float = 0.75
    prior: object = field(default=None, repr=False)
    _posteriors: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def fit(self, data_root, train_benchmarks: Sequence[str]):
        base = BayesianOffset().fit(data_root, train_benchmarks)
        episodes = {b: tag_episodes(str(data_root), b) for b in train_benchmarks}
        usable = sorted(b for b, e in episodes.items() if e)
        states, weights, eval_logits = [], [], []
        for inner_train, inner_held in benchmark_folds(usable, k=self.INNER_FOLDS, seed=2):
            prior = self.PRIOR_CLASS().fit(data_root, inner_train)
            for b in inner_held:
                for episode, (revealed, evaluated) in zip(
                        episodes[b], episode_logits(prior, episode_vectors(data_root, b), episodes[b])):
                    states.append(prior_states(episode, revealed, evaluated))
                    weights.append(episode["weight"])
                    eval_logits.append(evaluated)
        self.fit_states(states, weights, base.level, base.tau, float(np.mean(np.concatenate(eval_logits))))
        self.prior = self.PRIOR_CLASS().fit(data_root, train_benchmarks)
        return self

    def fit_states(self, states, weights, level: float, tau: float, center: float):
        """Grid (shift, scale) at the plain tau and sigma 0.75, then (tau, sigma) at the best pair."""
        self.level, self.center = float(level), float(center)
        stage_one = [(s, k) for s in self.SHIFTS for k in self.SCALES]
        scores = [prior_offset_alc(states, weights, level, s, k, center, tau, 0.75) for s, k in stage_one]
        self.shift, self.scale = (float(v) for v in stage_one[int(np.argmin(scores))])
        stage_two = [(tau * t, s) for t in self.TAU_SCALES for s in self.SIGMAS]
        scores = [prior_offset_alc(states, weights, level, self.shift, self.scale, center, t, s) for t, s in stage_two]
        self.tau, self.sigma = (float(v) for v in stage_two[int(np.argmin(scores))])
        self._posteriors.clear()
        return self

    def _logit(self, subject: dict, item: dict) -> float:
        base = self.level + self.shift
        if self.scale == 0 or self.prior is None:
            return base
        return base + self.scale * (self.prior.logit(subject, item) - self.center)

    def __call__(self, subject: dict, item: dict, labeled) -> float:
        examples = matching_examples(subject, item, labeled)
        key = (tuple(sorted(subject.items())), item.get("benchmark_id"),
               tuple((e.get("item_content"), e.get("item_features"), label) for e, label in examples))
        if key not in self._posteriors:
            if len(self._posteriors) >= self.CACHE_LIMIT:
                self._posteriors.clear()
            design = TagDesign.from_revealed([parse_tags(e.get("item_features", "")) for e, _ in examples])
            revealed = np.array([self._logit(subject, e) for e, _ in examples], dtype=float)
            mean, covariance = tag_offset_posterior(revealed, design.matrix, [y for _, y in examples],
                                                    self.tau, self.sigma)
            self._posteriors[key] = TagPosterior(design, mean, covariance)
        tags = parse_tags(item.get("item_features", ""))
        return self._posteriors[key].predict(tags, self._logit(subject, item), self.sigma)


class GBDTOffset(PriorOffset):
    """The same end-to-end model with gradient-boosted trees as the prior."""

    PRIOR_CLASS = GBDTPrior


__all__ = ["GBDTOffset", "PriorOffset", "episode_logits", "episode_vectors", "prior_offset_alc", "prior_states"]

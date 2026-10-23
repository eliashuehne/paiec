"""Small, weighted Unicode/structured ridge control for frozen-role experiments.

No data lookup or automatic fitting. Call fit with TRAINING inputs, labels and
equal-pair weights; keep whole families out for model selection.
"""
import hashlib
import json
import math
import re
import numpy as np

from .competition import competition_input
from .task_inputs import extract_features, task_sections

VERSION = "unicode_structured_ridge_v1"


def _hashed(text, dims, ngrams):
    vector = np.zeros(dims)
    for size in ngrams:
        for index in range(max(0,len(text)-size+1)):
            digest = hashlib.blake2s(text[index:index+size].encode(),digest_size=4).digest()
            value = int.from_bytes(digest,"little")
            vector[value % dims] += 1 if value & 0x80000000 else -1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def feature_vector(subject,item):
    subject,item=competition_input(subject,item)
    features=extract_features(subject,item)
    # Include both ends of long inputs. This is a cheap control with its own
    # declared feature limit, not an assertion of complete task visibility.
    text="\n".join(section[2] for _,section in task_sections(item["item_content"]))
    text=re.sub(r"\s+"," ",text.casefold())
    text=text if len(text)<=12000 else text[:6000]+text[-6000:]
    system=json.dumps(subject,sort_keys=True,ensure_ascii=False).casefold()
    task=features["task"]
    numeric=[task["characters"],task["cjk_characters"]/max(1,task["characters"]),
             task["option_count_hint"],task["structured"],task["interactors_recorded"],
             features["system"]["identity_unknown"]]
    numeric=[math.log1p(x) if x>1 else float(x) for x in numeric]
    return np.concatenate([_hashed(text,256,(2,3,4)),_hashed(system,64,(3,4)),numeric])


class FeatureRidge:
    name="feature_ridge"
    use_labels=False
    def __init__(self,l2=10.):
        if not math.isfinite(l2) or l2<=0:
            raise ValueError("l2 must be finite and positive")
        self.l2=l2
        self.weights=None
        self.runtime_metadata={"feature_control_version":VERSION,"l2":l2}

    def fit(self,inputs,labels,sample_weight):
        matrix=np.stack([feature_vector(*record) for record in inputs])
        labels=np.asarray(labels,dtype=float)
        weights=np.asarray(sample_weight,dtype=float)
        if (labels.shape!=(len(matrix),) or weights.shape!=labels.shape or
            not np.isin(labels,[0,1]).all() or not np.isfinite(weights).all() or (weights<=0).any()):
            raise ValueError("Binary labels and positive sample weights must align")
        weights=weights/weights.mean()
        self.center=np.average(matrix,axis=0,weights=weights)
        self.intercept=float(np.average(labels,weights=weights))
        centered=matrix-self.center
        gram=centered.T@(weights[:,None]*centered)+self.l2*np.eye(matrix.shape[1])
        self.weights=np.linalg.solve(gram,centered.T@(weights*(labels-self.intercept)))
        return self

    def __call__(self,subject,item,labeled):
        if self.weights is None:
            raise RuntimeError("Fit on training-role data before predicting")
        return float(np.clip(self.intercept+(feature_vector(subject,item)-self.center)@self.weights,0,1))

    def predict_many(self,subject,items,labeled):
        return [self(subject,item,[]) for item in items]

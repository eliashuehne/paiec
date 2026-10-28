"""Save the trained model's parameters and assemble the Codabench submission.

The zip holds ``model.py`` at its root, the numpy-only runtime modules it needs
(``features``, ``groups``, ``update``) under ``paiec_runtime/``, and the fitted
parameters in ``artifacts/prior_offset.npz``.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
ENTRY_POINT = ROOT / "submission" / "model.py"
RUNTIME_MODULES = ("features.py", "groups.py", "update.py")
ARTIFACT = Path("artifacts") / "prior_offset.npz"


def save_artifacts(model, path: Path) -> Path:
    """Write a fitted ``PriorOffset`` (logistic prior) as plain arrays and numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coef=np.asarray(model.prior.coef, dtype=float),
        intercept=float(model.prior.intercept),
        level=model.level, shift=model.shift, scale=model.scale, center=model.center,
        tau=model.tau, sigma=model.sigma,
    )
    return path


def assemble(target: Path, artifacts: Path) -> Path:
    """Lay out the submission directory exactly as it is zipped."""
    if target.exists():
        shutil.rmtree(target)
    runtime = target / "paiec_runtime"
    runtime.mkdir(parents=True)
    shutil.copy2(ENTRY_POINT, target / "model.py")
    (runtime / "__init__.py").write_text('"""Runtime modules copied from src/paiec."""\n')
    for name in RUNTIME_MODULES:
        shutil.copy2(PACKAGE / name, runtime / name)
    (target / ARTIFACT).parent.mkdir(parents=True)
    shutil.copy2(artifacts, target / ARTIFACT)
    return target


def build_zip(directory: Path, zip_path: Path) -> Path:
    """Zip the directory's contents with ``model.py`` at the root."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(directory).as_posix())
    return zip_path


__all__ = ["ARTIFACT", "assemble", "build_zip", "save_artifacts"]

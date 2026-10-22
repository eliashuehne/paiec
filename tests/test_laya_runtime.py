from pathlib import Path
from types import SimpleNamespace
import types
import unittest
from unittest.mock import Mock, patch

from paiec.laya_predictor import LayaPredictor
from paiec.laya_runtime import AgentRuntime, create_runtime


class NativeAgent:
    dtype = "float16"
    amp_enabled = True
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="float32")]))

    def predict_batch(self, states, questions, **kwargs):
        self.seen = (states, questions, kwargs)
        return [{"answers": {"correct": {"noul": .7}}} for _ in states]


def fake_mlx(**attributes):
    """A real `mlx`/`mlx.core` module pair, so `import mlx.core as mx` resolves."""
    package, core = types.ModuleType("mlx"), types.ModuleType("mlx.core")
    for name, value in attributes.items():
        setattr(core, name, value)
    package.core = core
    return {"mlx": package, "mlx.core": core}, core


class RuntimeTests(unittest.TestCase):
    def test_injected_backend_cannot_be_misreported(self):
        runtime = AgentRuntime(NativeAgent(), {"backend": "mlx", "device": "cpu", "model": "english", "checkpoint_revision": "test"})
        with self.assertRaisesRegex(ValueError, "differs from the requested backend"):
            LayaPredictor(backend="torch", router=runtime)

    def test_both_backends_receive_identical_shared_requests(self):
        seen = []
        for backend in ("torch", "mlx"):
            agent = NativeAgent()
            runtime = AgentRuntime(agent, {"backend": backend, "device": "cpu", "model": "english", "checkpoint_revision": "test", "compute_dtype": "test", "packages": {}})
            predictor = LayaPredictor(backend=backend, router=runtime, batch_size=2)
            self.assertEqual(predictor.predict_many({"normalized_name": "m"},
                                                   [{"item_content": "q"}, {"item_content": "other"}], []), [.7, .7])
            seen.append(agent.seen)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[0][2], {"batch_size": 2, "max_len": 512})

    def test_torch_does_not_import_mlx_and_uses_pinned_checkpoint(self):
        agent = NativeAgent()
        fake_laya = SimpleNamespace(load=Mock(return_value=agent))
        fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch.dict("sys.modules", {"torch": fake_torch, "laya": fake_laya, "laya_mlx": None}), \
                patch("paiec.laya_runtime.checkpoint_files", return_value=(Path("/checkpoint"), {"checkpoint_revision": "pinned"})) as checkpoint, \
                patch("paiec.laya_runtime.version", return_value="test"):
            runtime = create_runtime(backend="torch", model="english", device="auto", dtype="auto", revision="pinned", batch_size=2)
        checkpoint.assert_called_once_with("english", "pinned")
        fake_laya.load.assert_called_once_with("/checkpoint", device="mps")
        self.assertEqual(runtime.metadata["backend"], "torch")
        self.assertEqual(runtime.metadata["compute_dtype"], "upstream_auto")

    def test_mlx_does_not_import_torch_and_uses_same_pinned_checkpoint(self):
        modules, _ = fake_mlx(set_cache_limit=Mock(), set_memory_limit=Mock())
        fake_laya_mlx = SimpleNamespace(load=Mock(return_value=NativeAgent()))
        with patch.dict("sys.modules", {"laya_mlx": fake_laya_mlx, "torch": None, **modules}), \
                patch("paiec.laya_runtime.platform.system", return_value="Darwin"), \
                patch("paiec.laya_runtime.platform.machine", return_value="arm64"), \
                patch("paiec.laya_runtime.checkpoint_files", return_value=(Path("/checkpoint"), {"checkpoint_revision": "pinned"})) as checkpoint, \
                patch("paiec.laya_runtime.version", return_value="test"):
            runtime = create_runtime(backend="mlx", model="multilingual", device="auto", dtype="auto", revision="pinned", batch_size=8)
        checkpoint.assert_called_once_with("multilingual", "pinned")
        fake_laya_mlx.load.assert_called_once_with("/checkpoint", device="gpu", dtype="float16", batch_size=8, local_files_only=True)
        self.assertEqual(runtime.metadata["compute_dtype"], "float16")

    def test_unavailable_backend_fails_without_silent_fallback(self):
        with patch.dict("sys.modules", {"laya_mlx": None}), \
                patch("paiec.laya_runtime.platform.system", return_value="Darwin"), \
                patch("paiec.laya_runtime.platform.machine", return_value="arm64"):
            with self.assertRaisesRegex(RuntimeError, "MLX runtime is unavailable"):
                create_runtime(backend="mlx", model="english", device="auto", dtype="auto", revision=None, batch_size=1)

    def test_invalid_combinations_fail_before_loading(self):
        for backend, device, dtype in (("unknown", "auto", "auto"), ("mlx", "cuda", "auto"),
                                       ("torch", "cpu", "float16")):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                create_runtime(backend=backend, model="english", device=device, dtype=dtype, revision=None, batch_size=1)

    def test_mismatched_batch_or_closed_runtime_cannot_infer(self):
        runtime = AgentRuntime(NativeAgent(), {"model": "english"})
        request = {"model": "english", "state": "s", "questions": {}, "max_len": 512}
        with self.assertRaisesRegex(ValueError, "must share"):
            runtime.predict_batch([request, dict(request, max_len=1024)], batch_size=2)
        with self.assertRaisesRegex(ValueError, "loaded checkpoint"):
            runtime.predict_batch([dict(request, model="multilingual")], batch_size=1)
        runtime.agent = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.predict_batch([request], batch_size=1)

    def test_mlx_limits_are_set_before_the_checkpoint_loads(self):
        order = []
        modules, _ = fake_mlx(
            set_cache_limit=lambda value: order.append(("cache", value)),
            set_memory_limit=lambda value: order.append(("memory", value)),
        )
        fake_load = Mock(side_effect=lambda *a, **k: order.append(("load", None)) or NativeAgent())
        fake_laya_mlx = SimpleNamespace(load=fake_load)
        with patch.dict("sys.modules", {"laya_mlx": fake_laya_mlx, "torch": None, **modules}), \
                patch("paiec.laya_runtime.platform.system", return_value="Darwin"), \
                patch("paiec.laya_runtime.platform.machine", return_value="arm64"), \
                patch("paiec.laya_runtime.checkpoint_files", return_value=(Path("/checkpoint"), {"checkpoint_revision": "pinned"})), \
                patch("paiec.laya_runtime.version", return_value="test"):
            runtime = create_runtime(backend="mlx", model="english", device="auto", dtype="auto", revision=None, batch_size=8)
        self.assertEqual([name for name, _ in order], ["cache", "memory", "load"])
        self.assertEqual(runtime.metadata["mlx_cache_limit_bytes"], 2 * 1024**3)
        self.assertEqual(runtime.metadata["mlx_memory_limit_bytes"], 6 * 1024**3)

    def test_mlx_limits_accept_gib_overrides_and_reject_bad_ones(self):
        order = []
        modules, _ = fake_mlx(
            set_cache_limit=lambda value: order.append(("cache", value)),
            set_memory_limit=lambda value: order.append(("memory", value)),
        )
        with patch.dict("sys.modules", {"laya_mlx": SimpleNamespace(load=Mock(return_value=NativeAgent())),
                                        "torch": None, **modules}), \
                patch("paiec.laya_runtime.platform.system", return_value="Darwin"), \
                patch("paiec.laya_runtime.platform.machine", return_value="arm64"), \
                patch("paiec.laya_runtime.checkpoint_files", return_value=(Path("/checkpoint"), {"checkpoint_revision": "pinned"})), \
                patch("paiec.laya_runtime.version", return_value="test"), \
                patch.dict("os.environ", {"PAIEC_MLX_CACHE_LIMIT_GB": "1.5",
                                          "PAIEC_MLX_MEMORY_LIMIT_GB": "4"}):
            runtime = create_runtime(backend="mlx", model="english", device="auto", dtype="auto", revision=None, batch_size=1)
        self.assertEqual(dict(order), {"cache": int(1.5 * 1024**3), "memory": 4 * 1024**3})
        self.assertEqual(runtime.metadata["mlx_cache_limit_bytes"], int(1.5 * 1024**3))

        for bad in ("nonsense", "0", "-2"):
            with self.subTest(value=bad), patch.dict("os.environ", {"PAIEC_MLX_CACHE_LIMIT_GB": bad}), \
                    patch.dict("sys.modules", {"laya_mlx": SimpleNamespace(load=Mock(return_value=NativeAgent())),
                                               "torch": None, **fake_mlx()[0]}), \
                    patch("paiec.laya_runtime.platform.system", return_value="Darwin"), \
                    patch("paiec.laya_runtime.platform.machine", return_value="arm64"):
                with self.assertRaisesRegex(ValueError, "PAIEC_MLX_CACHE_LIMIT_GB"):
                    create_runtime(backend="mlx", model="english", device="auto", dtype="auto", revision=None, batch_size=1)

    def test_torch_never_imports_mlx_or_touches_its_limits(self):
        asked = []
        modules, _ = fake_mlx(set_cache_limit=asked.append, set_memory_limit=asked.append)
        fake_laya = SimpleNamespace(load=Mock(return_value=NativeAgent()))
        fake_torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch.dict("sys.modules", {"torch": fake_torch, "laya": fake_laya,
                                        "laya_mlx": None, **modules}), \
                patch("paiec.laya_runtime.checkpoint_files", return_value=(Path("/checkpoint"), {"checkpoint_revision": "pinned"})), \
                patch("paiec.laya_runtime.version", return_value="test"):
            create_runtime(backend="torch", model="english", device="auto", dtype="auto", revision=None, batch_size=1)
        self.assertEqual(asked, [])

    def test_trim_cache_releases_backend_allocations_only(self):
        calls = []
        modules, _ = fake_mlx(clear_cache=lambda: calls.append("mlx_clear"),
                              synchronize=lambda: calls.append("mlx_sync"))
        agent = NativeAgent()
        with patch.dict("sys.modules", modules):
            runtime = AgentRuntime(agent, {"backend": "mlx", "device": "gpu", "model": "english"})
            runtime.trim_cache()
        self.assertEqual(calls, ["mlx_sync", "mlx_clear"])
        self.assertIs(runtime.agent, agent)
        runtime.agent = None
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.predict_batch([{"model": "english", "state": "s", "questions": {}, "max_len": 512}], batch_size=1)

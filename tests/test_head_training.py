import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from paiec.head_training import HeadSmokeConfig, file_hash, select_smoke_rows
from paiec.laya_runtime import local_checkpoint_files


class HeadSmokeTests(unittest.TestCase):
    def test_selection_is_training_only_label_blind_and_pair_balanced(self):
        rows = []
        for subject, count in [('a', 6), ('b', 2)]:
            for i in range(count):
                rows.append(dict(role='training', exported=True, truncated=False,
                                 benchmark='bench', subject_id=subject, jsonl_line=len(rows), outcome=i % 2))
        rows.extend([dict(rows[0], role='confirmation'), dict(rows[0], truncated=True),
                     dict(rows[0], exported=False)])
        frame = pd.DataFrame(rows)
        chosen = select_smoke_rows(frame, 6, 0)
        altered = select_smoke_rows(frame.assign(outcome=1-frame.outcome), 6, 0)
        self.assertEqual([r['jsonl_line'] for r in chosen], [r['jsonl_line'] for r in altered])
        self.assertTrue(all(r['role'] == 'training' and not r['truncated'] and r['exported'] for r in chosen))
        weights = pd.DataFrame(chosen).groupby('subject_id').smoke_weight.sum()
        self.assertAlmostEqual(weights.a, weights.b)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_smoke_rows(frame, 9, 0)

    def test_budget_cannot_expand_beyond_authorized_smoke(self):
        for settings in ({'updates': 11}, {'samples': 257}, {'max_len': 1024}):
            with self.assertRaises(ValueError):
                HeadSmokeConfig(**settings)

    def test_local_checkpoint_identity_requires_integrity_and_matching_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            names = ['model.safetensors', 'rl_agent_config.json', 'encoder/config.json', 'tokenizer/tokenizer.json']
            for name in names:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('{}')
            manifest = {'model': 'english', 'artifact_sha256': {n: file_hash(root/n) for n in names}}
            (root/'paiec_training.json').write_text(json.dumps(manifest))
            _, source = local_checkpoint_files(folder, 'english')
            self.assertTrue(source['checkpoint_revision'].startswith('local-'))
            with self.assertRaisesRegex(ValueError, 'differs'):
                local_checkpoint_files(folder, 'multilingual')
            (root/'model.safetensors').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                local_checkpoint_files(folder, 'english')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch reference environment only')
    def test_weighted_microbatch_gradients_match_whole_window(self):
        import torch
        from paiec.head_training import weighted_ce
        logits = torch.tensor([[.2, .3], [.8, -.1], [.3, .4]], requires_grad=True)
        target = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        weights = torch.tensor([1., 3., 2.])
        denominator = weights.sum()
        weighted_ce(logits, target, weights, denominator).backward()
        expected = logits.grad.clone()
        logits.grad = None
        weighted_ce(logits[:2], target[:2], weights[:2], denominator).backward()
        weighted_ce(logits[2:], target[2:], weights[2:], denominator).backward()
        torch.testing.assert_close(logits.grad, expected)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch reference environment only')
    def test_only_decision_modules_update_and_encoder_dropout_stays_off(self):
        import torch
        from paiec.head_training import freeze_for_head, tensor_hashes
        model = torch.nn.Module()
        model.encoder = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.9))
        model.head = torch.nn.Linear(2, 2)
        model.type_emb = torch.nn.Embedding(3, 2)
        model.scorer = torch.nn.Linear(2, 1)
        model.act_head = torch.nn.Linear(2, 1)
        before = tensor_hashes(model)
        params = freeze_for_head(model)
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.head.training)
        optimizer = torch.optim.AdamW(params, lr=.01)
        hidden = model.encoder(torch.ones(1, 2)) + model.type_emb(torch.tensor([0]))
        model.scorer(model.head(hidden)).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.act_head.parameters()))
        optimizer.step()
        after = tensor_hashes(model)
        self.assertEqual(before['encoder'], after['encoder'])
        self.assertEqual(before['act_head'], after['act_head'])
        self.assertNotEqual(before['decision'], after['decision'])

    @unittest.skipUnless(importlib.util.find_spec('laya'), 'PyTorch reference environment only')
    def test_export_normalization_precedes_hashing_and_is_stable_on_reload(self):
        from paiec.head_training import normalize_tokenizer_export
        from laya.agent import _fix_tokenizer_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer').mkdir()
            config = root/'tokenizer/tokenizer_config.json'
            config.write_text(json.dumps({'tokenizer_class':'TokenizersBackend',
                                          'backend':'tokenizers', 'extra_special_tokens':['<x>']}))
            normalize_tokenizer_export(root)
            expected = file_hash(config)
            _fix_tokenizer_config(str(root))
            self.assertEqual(file_hash(config), expected)
            self.assertEqual(json.loads(config.read_text())['tokenizer_class'], 'PreTrainedTokenizerFast')

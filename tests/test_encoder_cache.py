import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch cache checks")
class EncoderCacheTests(unittest.TestCase):
    def encoder(self):
        import torch
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(20, 4)
                self.config = SimpleNamespace(hidden_size=4)
                self.calls = 0
            def forward(self, input_ids, attention_mask):
                self.calls += 1
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        encoder = Encoder().eval()
        encoder.requires_grad_(False)
        return encoder

    def test_repadding_reordering_eviction_and_export_keys(self):
        import torch
        from paiec.encoder_cache import FrozenEncoderCache
        encoder = self.encoder()
        keys = list(encoder.state_dict())
        cache = FrozenEncoderCache(encoder, 64, identity="fixture")
        ids = torch.tensor([[1,2,0],[3,0,0]])
        mask = torch.tensor([[1,1,0],[1,0,0]])
        old = cache.original(ids,mask).last_hidden_state
        first = encoder(ids,mask).last_hidden_state
        self.assertTrue(torch.equal(old[mask.bool()],first[mask.bool()]))
        calls = encoder.calls
        new_ids = torch.tensor([[3,0,0,0],[1,2,0,0]])
        new_mask = torch.tensor([[1,0,0,0],[1,1,0,0]])
        second = encoder(new_ids,new_mask).last_hidden_state
        self.assertEqual(encoder.calls,calls)
        self.assertTrue(torch.equal(second[1,:2],first[0,:2]))
        encoder(torch.tensor([[4,5,6,7]]),torch.ones(1,4,dtype=torch.long))
        self.assertLessEqual(cache.bytes,64)
        self.assertGreater(cache.evictions,0)
        self.assertEqual(list(encoder.state_dict()),keys)
        cache.close()
        self.assertEqual(cache.bytes,0)
        encoder(ids,mask)

    def test_training_weight_changes_and_invalid_masks_fail_closed(self):
        import torch
        from paiec.encoder_cache import FrozenEncoderCache
        encoder = self.encoder()
        cache = FrozenEncoderCache(encoder,1024,identity="fixture")
        ids = torch.tensor([[1,2,0]])
        with self.assertRaises(ValueError):
            encoder(ids,torch.tensor([[1,0,1]]))
        encoder.train()
        with self.assertRaises(RuntimeError):
            encoder(ids,torch.tensor([[1,1,0]]))
        encoder.eval()
        with torch.no_grad():
            encoder.embedding.weight.add_(1)
        with self.assertRaises(RuntimeError):
            encoder(ids,torch.tensor([[1,1,0]]))
        cache.close()

    def test_native_head_probabilities_gradients_rng_and_save_load(self):
        import torch
        from laya.common import DecisionModel
        from paiec.encoder_cache import FrozenEncoderCache,validate_cache
        from paiec.head_training import freeze_for_head
        encoder = self.encoder()
        model = DecisionModel(encoder,head_layers=1)
        freeze_for_head(model)
        batch = dict(input_ids=torch.tensor([[1,2,3],[4,5,0]]),
            attention_mask=torch.tensor([[1,1,1],[1,1,0]]),
            marker_pos=torch.tensor([[0,1],[0,1]]),marker_mask=torch.ones(2,2,dtype=torch.bool),
            qtype=torch.tensor([2,2]))
        cache = FrozenEncoderCache(encoder,1024,identity="fixture")
        rng = torch.get_rng_state()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        report = validate_cache(model,cache,[batch],torch.device("cpu"))
        self.assertLessEqual(report["max_gradient_difference"],1e-6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        cache.close()
        model.load_state_dict(before)

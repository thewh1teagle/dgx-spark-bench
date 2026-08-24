"""renikud training-step benchmark: real architecture, dummy data.

Mirrors renikud's actual train step -- dicta-il/dictabert-large-char encoder plus
three coupled classification heads, fp16 autocast + GradScaler, AdamW with
discriminative LRs. Uses random batches, so no dataset is needed and it runs
identically on any box.

Requires: transformers (downloads ~700MB encoder on first run).
"""

import time

import torch
import torch.nn as nn

# From renikud src/phonology.py and src/constants.py
NUM_CONSONANT, NUM_VOWEL, NUM_STRESS = 25, 6, 2
MAX_LEN = 256
ENCODER_ID = "dicta-il/dictabert-large-char"


class G2PModel(nn.Module):
    """Mirrors renikud src/model.py: encoder + three coupled heads."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.consonant_head = nn.Linear(h, NUM_CONSONANT)
        self.vowel_head = nn.Linear(h + NUM_CONSONANT, NUM_VOWEL)
        self.stress_head = nn.Linear(h + NUM_CONSONANT + NUM_VOWEL, NUM_STRESS)

    def forward(self, ids, mask, labels):
        hidden = self.dropout(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state)
        c = self.consonant_head(hidden)
        v = self.vowel_head(torch.cat([hidden, c], dim=-1))
        s = self.stress_head(torch.cat([hidden, c, v], dim=-1))
        ce = nn.CrossEntropyLoss()
        cl, vl, sl = labels
        return (ce(c.view(-1, NUM_CONSONANT), cl.view(-1))
                + ce(v.view(-1, NUM_VOWEL), vl.view(-1))
                + ce(s.view(-1, NUM_STRESS), sl.view(-1)))


def load_model():
    from transformers import AutoModel
    return G2PModel(AutoModel.from_pretrained(ENCODER_ID)).cuda()


def bench(batch=16, seq=MAX_LEN, iters=10, warmup=3, fp16=True, model=None):
    """Time renikud's train step. Returns None on OOM.

    Pass `model` to reuse a already-loaded encoder across a sweep -- reloading
    the 700MB checkpoint per batch size dominates sweep wall-clock otherwise.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = model or load_model()
        # renikud src/optimizer.py: discriminative LRs, encoder vs heads
        heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        opt = torch.optim.AdamW(
            [{"params": enc, "lr": 2e-5}, {"params": heads, "lr": 1e-4}], weight_decay=0.01
        )
        scaler = torch.amp.GradScaler("cuda", enabled=fp16)

        vocab = model.encoder.config.vocab_size
        ids = torch.randint(0, vocab, (batch, seq), device="cuda")
        mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        labels = tuple(
            torch.randint(0, n, (batch, seq), device="cuda")
            for n in (NUM_CONSONANT, NUM_VOWEL, NUM_STRESS)
        )

        def step():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(ids, mask, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        s = sorted(times)[len(times) // 2]

        peak = torch.cuda.max_memory_allocated() / 1e9
        params = sum(p.numel() for p in model.parameters())
        model.zero_grad(set_to_none=True)
        del opt, ids, mask, labels
        torch.cuda.empty_cache()
        return {
            "params_m": round(params / 1e6),
            "batch": batch, "seq": seq, "precision": "fp16" if fp16 else "fp32",
            "it_s": round(1 / s, 2),
            "samples_s": round(batch / s, 1),
            "peak_vram_gb": round(peak, 1),
            "measured": True,
        }
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

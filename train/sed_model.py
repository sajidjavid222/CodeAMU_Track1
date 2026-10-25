"""CRNN sound-event-detection model for Track 1 (binary noise vs. no-noise).

Front end : 64-band log-mel, 25 ms window / 10 ms hop (10 ms hop is chosen to
            line up with the 10 ms Segment-Dice grid).
Encoder   : 3 conv blocks. Frequency is pooled away entirely; time is pooled
            by 2 only, so the model emits one frame every 20 ms. That is fine
            against the metric's 50 ms tolerance floor while costing ~half the
            sequence length.
Recurrent : 2-layer BiGRU over time.
Heads     : - strong  : per-frame binary logit (the actual detection output)
            - cat     : per-frame 7-way category logits (auxiliary, shapes the
                        representation; not submitted)
            - weak    : attention-pooled clip-level logits, which is how the
                        Bronze tier (tags but no timestamps) is trained.

The weak head is a linear-softmax attention pool, the standard DCASE
weakly-supervised formulation: clip probability is a learned attention-weighted
average of frame probabilities, so gradient from a clip-level tag flows into
the frames that caused it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

N_CATEGORIES = 7
SR = 16000
HOP = 160          # 10 ms
WIN = 400          # 25 ms
N_MELS = 64
# Model frame size. 2 -> 20 ms (default, what every existing checkpoint uses);
# 1 -> 10 ms. Set SED_TIME_POOL=1 in the environment BEFORE importing this
# module -- sed_data derives FRAME_SEC from it at import time, so it cannot be
# changed afterwards. Event-F1 judges 14% of events at a 50 ms tolerance floor
# while the Gaussian boundary target is 2 frames wide, i.e. 40 ms at TIME_POOL=2:
# the training target is blurrier than the tolerance it is scored against.
TIME_POOL = int(os.environ.get("SED_TIME_POOL", "2"))


class LogMel(nn.Module):
    def __init__(self, sr=SR, n_mels=N_MELS, win=WIN, hop=HOP):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, win_length=win, hop_length=hop,
            f_min=20.0, f_max=sr // 2, n_mels=n_mels, power=2.0, center=True)

    def forward(self, wav):                      # (B, T) -> (B, n_mels, F)
        return torch.log(self.mel(wav) + 1e-6)


class SpecAugment(nn.Module):
    """Frequency / time masking. Time masks are kept short so frame targets
    stay meaningful for the strongly-labelled tiers."""

    def __init__(self, n_freq=2, freq_w=12, n_time=2, time_w=20):
        super().__init__()
        self.n_freq, self.freq_w = n_freq, freq_w
        self.n_time, self.time_w = n_time, time_w

    def forward(self, x):                        # (B, n_mels, F)
        if not self.training:
            return x
        B, M, T = x.shape
        for _ in range(self.n_freq):
            w = torch.randint(0, self.freq_w + 1, (1,)).item()
            if w:
                f0 = torch.randint(0, max(1, M - w), (1,)).item()
                x[:, f0:f0 + w, :] = 0
        for _ in range(self.n_time):
            w = torch.randint(0, self.time_w + 1, (1,)).item()
            if w and T > w:
                t0 = torch.randint(0, T - w, (1,)).item()
                x[:, :, t0:t0 + w] = 0
        return x


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool_f, pool_t, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.pool = nn.AvgPool2d((pool_f, pool_t))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.drop(self.pool(x))


class CRNN(nn.Module):
    def __init__(self, n_mels=N_MELS, channels=(32, 64, 128), rnn_hidden=128,
                 rnn_layers=2, dropout=0.2, n_categories=N_CATEGORIES):
        super().__init__()
        self.logmel = LogMel(n_mels=n_mels)
        self.specaug = SpecAugment()
        self.in_norm = nn.BatchNorm2d(1)

        # Frequency is pooled all the way to 1; time only by TIME_POOL (20 ms).
        # The per-block frequency factors must multiply to n_mels, so they are
        # derived rather than hardcoded -- a 3-block/64-mel and a 4-block/128-mel
        # configuration both have to land on exactly one frequency bin.
        nb = len(channels)
        fpool = [4] * nb
        prod = 4 ** nb
        if prod != n_mels:                       # fix up the last factor
            last = n_mels // (4 ** (nb - 1))
            if last < 1 or (4 ** (nb - 1)) * last != n_mels:
                raise ValueError(f"cannot pool {n_mels} mels with {nb} blocks")
            fpool[-1] = last
        cin = [1] + list(channels[:-1])
        self.blocks = nn.ModuleList([
            ConvBlock(cin[i], channels[i], fpool[i], TIME_POOL if i == 1 else 1)
            for i in range(nb)
        ])
        self.rnn = nn.GRU(channels[-1], rnn_hidden, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        d = rnn_hidden * 2
        self.drop = nn.Dropout(dropout)
        self.strong = nn.Linear(d, 1)
        self.cat = nn.Linear(d, n_categories)
        self.att = nn.Linear(d, 1)               # attention for weak pooling
        self.att_cat = nn.Linear(d, n_categories)
        # Dedicated boundary heads. Event-F1 scores edge placement (median
        # tolerance on gold is only ~89 ms), and a presence-only head is never
        # asked to be sharp at the edges -- it is rewarded for covering the
        # middle. These predict Gaussian bumps at onsets/offsets so the loss
        # supervises the boundaries themselves.
        self.onset = nn.Linear(d, 1)
        self.offset = nn.Linear(d, 1)

    def forward(self, wav, frame_mask=None):
        """wav: (B, T) float32 at 16 kHz.

        Returns dict with
          strong_logit (B, F'), cat_logit (B, F', C),
          weak_logit (B,), weak_cat_logit (B, C)
        where F' is the number of 20 ms frames.
        """
        x = self.logmel(wav)                     # (B, M, F)
        x = self.specaug(x)
        x = self.in_norm(x.unsqueeze(1))         # (B, 1, M, F)
        for b in self.blocks:
            x = b(x)
        x = x.mean(dim=2).transpose(1, 2)        # (B, F', C)
        x, _ = self.rnn(x)
        x = self.drop(x)

        strong = self.strong(x).squeeze(-1)      # (B, F')
        cat = self.cat(x)                        # (B, F', C)

        att = self.att(x).squeeze(-1)            # (B, F')
        if frame_mask is not None:
            m = frame_mask[:, :att.shape[1]]
            att = att.masked_fill(~m, torch.finfo(att.dtype).min)
            strong = strong.masked_fill(~m, torch.finfo(strong.dtype).min)
        w = torch.softmax(att, dim=-1)
        weak = (w * strong.clamp(min=-30)).sum(-1)

        acat = self.att_cat(x)
        if frame_mask is not None:
            m = frame_mask[:, :acat.shape[1]].unsqueeze(-1)
            acat = acat.masked_fill(~m, torch.finfo(acat.dtype).min)
        wc = torch.softmax(acat, dim=1)
        weak_cat = (wc * cat).sum(1)

        return {"strong_logit": strong, "cat_logit": cat,
                "weak_logit": weak, "weak_cat_logit": weak_cat,
                "onset_logit": self.onset(x).squeeze(-1),
                "offset_logit": self.offset(x).squeeze(-1)}

    @staticmethod
    def n_frames(n_samples):
        """20 ms model frames produced for a waveform of `n_samples`."""
        return (n_samples // HOP + 1) // TIME_POOL

    @property
    def frame_sec(self):
        return HOP * TIME_POOL / SR              # 0.02 s


class SSLSED(nn.Module):
    """Pretrained speech-SSL encoder + BiGRU detector head.

    WavLM is the default encoder and is a deliberate choice for *this* task:
    its pretraining objective simulates noisy and overlapped speech, so its
    representations already separate "speech" from "interfering non-speech" --
    which is exactly the decision boundary Track 1 scores. It also emits
    ~20.1 ms frames, matching both the CRNN branch and the 50 ms tolerance
    floor, so no resampling is needed to ensemble the two.

    The 13 hidden states are combined by a learnable weighted sum (the SUPERB
    recipe) rather than taking the last layer: lower layers carry the acoustic
    detail that boundary precision needs, upper layers carry event semantics.
    """

    def __init__(self, ssl_name="microsoft/wavlm-base-plus", rnn_hidden=256,
                 rnn_layers=2, dropout=0.2, n_categories=N_CATEGORIES,
                 freeze_feature_extractor=True, layer_sum=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.ssl_name = ssl_name
        self.cfg = AutoConfig.from_pretrained(ssl_name)
        self.encoder = AutoModel.from_pretrained(ssl_name)
        if freeze_feature_extractor and hasattr(self.encoder, "freeze_feature_encoder"):
            self.encoder.freeze_feature_encoder()      # conv frontend stays fixed

        n_layers = self.cfg.num_hidden_layers + 1
        self.layer_sum = layer_sum
        if layer_sum:
            self.layer_w = nn.Parameter(torch.zeros(n_layers))

        d_in = self.cfg.hidden_size
        self.rnn = nn.GRU(d_in, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        d = rnn_hidden * 2
        self.drop = nn.Dropout(dropout)
        self.strong = nn.Linear(d, 1)
        self.cat = nn.Linear(d, n_categories)
        self.att = nn.Linear(d, 1)
        self.att_cat = nn.Linear(d, n_categories)
        self.onset = nn.Linear(d, 1)             # see CRNN: boundary heads
        self.offset = nn.Linear(d, 1)

    @property
    def frame_sec(self):
        return 0.02

    def forward(self, wav, frame_mask=None):
        out = self.encoder(wav, output_hidden_states=self.layer_sum)
        if self.layer_sum:
            hs = torch.stack(out.hidden_states, dim=0)      # (L, B, F, D)
            w = torch.softmax(self.layer_w, dim=0).view(-1, 1, 1, 1)
            x = (hs * w).sum(0)
        else:
            x = out.last_hidden_state

        x, _ = self.rnn(x)
        x = self.drop(x)

        strong = self.strong(x).squeeze(-1)
        cat = self.cat(x)

        att = self.att(x).squeeze(-1)
        if frame_mask is not None:
            # The SSL encoder's frame count rarely matches the mask's exactly,
            # so pad/truncate the mask to the emitted length; frames beyond the
            # mask are padding and must not win attention.
            m = frame_mask[:, :att.shape[1]]
            if m.shape[1] < att.shape[1]:
                pad = torch.zeros(m.shape[0], att.shape[1] - m.shape[1],
                                  dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=1)
            att = att.masked_fill(~m, torch.finfo(att.dtype).min)
            strong = strong.masked_fill(~m, torch.finfo(strong.dtype).min)
        w = torch.softmax(att, dim=-1)
        weak = (w * strong.clamp(min=-30)).sum(-1)

        acat = self.att_cat(x)
        wc = torch.softmax(acat, dim=1)
        weak_cat = (wc * cat).sum(1)

        return {"strong_logit": strong, "cat_logit": cat,
                "weak_logit": weak, "weak_cat_logit": weak_cat,
                "onset_logit": self.onset(x).squeeze(-1),
                "offset_logit": self.offset(x).squeeze(-1)}

    def param_groups(self, lr_head=1e-3, lr_encoder=3e-5):
        """Encoder needs a much smaller LR than the freshly-initialised head,
        or the first epochs wash out the pretrained features."""
        enc, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith("encoder.") else head).append(p)
        return [{"params": enc, "lr": lr_encoder},
                {"params": head, "lr": lr_head}]


def build_model(name="crnn", **kw):
    if name == "crnn":
        return CRNN(**kw)
    if name == "crnn_big":
        # A scaled-up convolutional member. Every other model in the ensemble is
        # a fine-tune of a SPEECH-pretrained transformer (Whisper / WavLM), yet
        # the 0.79M-parameter CRNN earns the same ensemble weight as
        # whisper-large in the OOF weight search -- convolution-on-mel is the
        # one genuinely complementary inductive bias in the pool, and it is the
        # only family that was never scaled. 128 mels, four conv blocks, GRU-256.
        # Frequency is pooled 4*4*4*2 = 128 -> 1 and time only by TIME_POOL, so
        # it still emits 20 ms frames and drops into the existing ensemble.
        kw.setdefault("n_mels", 128)
        kw.setdefault("channels", (64, 128, 256, 256))
        kw.setdefault("rnn_hidden", 256)
        kw.setdefault("rnn_layers", 2)
        kw.setdefault("dropout", 0.25)
        return CRNN(**kw)
    if name in ("ssl", "wavlm"):
        return SSLSED(**kw)
    if name == "whisper":
        return WhisperSED(**kw)
    if name == "beats_hybrid":
        return BeatsHybridSED(**kw)
    raise ValueError(f"unknown model {name!r}")


if __name__ == "__main__":
    m = CRNN()
    n = sum(p.numel() for p in m.parameters())
    wav = torch.randn(2, SR * 3)
    out = m(wav)
    print(f"params={n/1e6:.2f}M  frame_sec={m.frame_sec}")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("expected frames for 3 s:", CRNN.n_frames(SR * 3))


class WhisperSED(nn.Module):
    """Whisper encoder (680k h of real-world audio) as a frame-level detector.

    Two things make Whisper attractive here: it emits exactly 20 ms frames --
    the resolution this metric needs -- and it was pretrained on genuinely
    noisy in-the-wild audio rather than clean speech corpora.

    The stock `WhisperEncoder.forward` hard-asserts 3000 mel frames (30 s), so
    a 7 s clip would cost 4x the compute it needs. This calls conv1/conv2, the
    layer stack and the final norm directly with positional embeddings sliced
    to the true length, which is what the assertion is protecting rather than
    something the maths requires.
    """

    def __init__(self, ssl_name="openai/whisper-small", rnn_hidden=256,
                 rnn_layers=2, dropout=0.2, n_categories=N_CATEGORIES,
                 freeze_feature_extractor=True, layer_sum=True):
        super().__init__()
        from transformers import WhisperModel, WhisperFeatureExtractor
        import numpy as np

        self.ssl_name = ssl_name
        # force fp32: whisper-large-v3 ships fp16 weights, and AMP's
        # GradScaler cannot unscale fp16 gradients -- the run dies with
        # "Attempting to unscale FP16 gradients" the moment the encoder
        # unfreezes, leaving only the frozen warm-up epoch.
        full = WhisperModel.from_pretrained(ssl_name, torch_dtype=torch.float32)
        self.encoder = full.encoder
        del full
        self.cfg = self.encoder.config
        if freeze_feature_extractor:
            for p in list(self.encoder.conv1.parameters()) + \
                     list(self.encoder.conv2.parameters()):
                p.requires_grad_(False)

        fe = WhisperFeatureExtractor.from_pretrained(ssl_name)
        filters = torch.tensor(np.array(fe.mel_filters), dtype=torch.float32).T
        self.register_buffer("mel_filters", filters)          # (80, 201)
        self.register_buffer("win", torch.hann_window(400))

        self.layer_sum = layer_sum
        if layer_sum:
            self.layer_w = nn.Parameter(torch.zeros(self.cfg.encoder_layers + 1))

        d_in = self.cfg.d_model
        self.rnn = nn.GRU(d_in, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        d = rnn_hidden * 2
        self.drop = nn.Dropout(dropout)
        self.strong = nn.Linear(d, 1)
        self.cat = nn.Linear(d, n_categories)
        self.att = nn.Linear(d, 1)
        self.att_cat = nn.Linear(d, n_categories)
        self.onset = nn.Linear(d, 1)
        self.offset = nn.Linear(d, 1)

    @property
    def frame_sec(self):
        return 0.02

    def log_mel(self, wav):
        """Whisper's exact log-mel (verified against WhisperFeatureExtractor)."""
        stft = torch.stft(wav, 400, 160, window=self.win.to(wav.device),
                          return_complex=True)
        mag = stft[..., :-1].abs() ** 2
        mel = self.mel_filters.to(wav.device) @ mag
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(
            log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0

    def encode(self, mel):
        e = self.encoder
        x = F.gelu(e.conv1(mel))
        x = F.gelu(e.conv2(x)).permute(0, 2, 1)          # (B, T/2, d)
        x = x + e.embed_positions.weight[:x.shape[1]]     # slice, don't pad
        hs = [x]
        for layer in e.layers:
            out = layer(x, None)
            x = out[0] if isinstance(out, tuple) else out
            hs.append(x)
        x = e.layer_norm(x)
        hs[-1] = x
        return x, hs

    def forward(self, wav, frame_mask=None):
        mel = self.log_mel(wav)
        if mel.shape[-1] % 2:                              # conv2 stride 2
            mel = mel[..., :-1]
        x, hs = self.encode(mel)
        if self.layer_sum:
            w = torch.softmax(self.layer_w, dim=0).view(-1, 1, 1, 1)
            x = (torch.stack(hs, 0) * w).sum(0)

        # the Whisper encoder is fixed at 20 ms; lift it onto the model grid
        up = 2 // TIME_POOL
        if up > 1:
            x = F.interpolate(x.transpose(1, 2), scale_factor=up,
                              mode="linear", align_corners=False).transpose(1, 2)

        x, _ = self.rnn(x)
        x = self.drop(x)
        strong = self.strong(x).squeeze(-1)
        cat = self.cat(x)
        att = self.att(x).squeeze(-1)
        if frame_mask is not None:
            m = frame_mask[:, :att.shape[1]]
            if m.shape[1] < att.shape[1]:
                pad = torch.zeros(m.shape[0], att.shape[1] - m.shape[1],
                                  dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=1)
            att = att.masked_fill(~m, torch.finfo(att.dtype).min)
            strong = strong.masked_fill(~m, torch.finfo(strong.dtype).min)
        w = torch.softmax(att, dim=-1)
        weak = (w * strong.clamp(min=-30)).sum(-1)
        acat = self.att_cat(x)
        weak_cat = (torch.softmax(acat, dim=1) * cat).sum(1)
        return {"strong_logit": strong, "cat_logit": cat,
                "weak_logit": weak, "weak_cat_logit": weak_cat,
                "onset_logit": self.onset(x).squeeze(-1),
                "offset_logit": self.offset(x).squeeze(-1)}

    def param_groups(self, lr_head=1e-3, lr_encoder=3e-5):
        enc, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith("encoder.") else head).append(p)
        return [{"params": enc, "lr": lr_encoder},
                {"params": head, "lr": lr_head}]


class BeatsHybridSED(nn.Module):
    """DCASE-style hybrid: fine-resolution Whisper branch + frozen BEATs semantics.

    BEATs is the AudioSet SOTA encoder, but its patch grid gives a TRUE time
    resolution of ~161 ms -- far coarser than this metric's 89 ms median
    tolerance, so it cannot localise events on its own. (Its `extract_features`
    output looks like 20 ms only because a T'xF' grid is flattened; pooling over
    frequency recovers the real 62 steps per 10 s.)

    The DCASE recipe is therefore to use it as a *semantic* side-branch:
    frequency-pool it, upsample x8 onto the fine grid, and concatenate. Whisper
    says precisely WHEN something happens; BEATs says WHAT it is. BEATs stays
    frozen -- it is 91M of AudioSet knowledge we have no data to improve.
    """

    def __init__(self, ssl_name="openai/whisper-medium", beats_ckpt=None,
                 rnn_hidden=256, rnn_layers=2, dropout=0.2,
                 n_categories=N_CATEGORIES, freeze_beats=True):
        super().__init__()
        self.fine = WhisperSED(ssl_name=ssl_name, rnn_hidden=rnn_hidden,
                               rnn_layers=rnn_layers, dropout=dropout,
                               n_categories=n_categories)
        # reuse the Whisper front end, but replace the head with a joint one
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "beats"))
        from beats.BEATs import BEATs, BEATsConfig
        ck = torch.load(beats_ckpt, map_location="cpu", weights_only=False)
        self.beats = BEATs(BEATsConfig(ck["cfg"]))
        self.beats.load_state_dict(ck["model"], strict=False)
        self.beats.predictor = None                 # we want features, not AudioSet logits
        self.beats_dim = ck["cfg"]["encoder_embed_dim"]
        if freeze_beats:
            for p in self.beats.parameters():
                p.requires_grad_(False)
            self.beats.eval()

        d_fine = self.fine.cfg.d_model
        self.rnn = nn.GRU(d_fine + self.beats_dim, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        d = rnn_hidden * 2
        self.drop = nn.Dropout(dropout)
        self.strong = nn.Linear(d, 1)
        self.cat = nn.Linear(d, n_categories)
        self.att = nn.Linear(d, 1)
        self.att_cat = nn.Linear(d, n_categories)
        self.onset = nn.Linear(d, 1)
        self.offset = nn.Linear(d, 1)

    def train(self, mode=True):
        """Parent .train() must not re-enable dropout inside frozen BEATs."""
        super().train(mode)
        self.beats.eval()
        return self

    @property
    def frame_sec(self):
        return 0.02

    def beats_features(self, wav, n_target):
        """BEATs tokens -> frequency-pooled, upsampled to the fine grid.

        Forced to fp32: under AMP the kaldi-fbank preprocess and the weight_norm
        encoder produce NaNs from the first step (the branch is frozen, so fp32
        costs no extra memory for gradients).
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            wav = wav.float()
            fb = self.beats.preprocess(wav)                 # (B, T, 128)
            emb = self.beats.patch_embedding(fb.unsqueeze(1))   # (B, C, T', F')
            B, C, T2, F2 = emb.shape
            x = emb.reshape(B, C, T2 * F2).transpose(1, 2)
            x = self.beats.layer_norm(x)
            if self.beats.post_extract_proj is not None:
                x = self.beats.post_extract_proj(x)
            x, _ = self.beats.encoder(self.beats.dropout_input(x))
            x = x.reshape(B, T2, F2, -1).mean(2)            # pool frequency -> (B, T', D)
        x = x.transpose(1, 2)                                # (B, D, T')
        x = F.interpolate(x, size=n_target, mode="linear", align_corners=False)
        return x.transpose(1, 2)                             # (B, n_target, D)

    def forward(self, wav, frame_mask=None):
        mel = self.fine.log_mel(wav)
        if mel.shape[-1] % 2:
            mel = mel[..., :-1]
        f, hs = self.fine.encode(mel)
        if self.fine.layer_sum:
            w = torch.softmax(self.fine.layer_w, dim=0).view(-1, 1, 1, 1)
            f = (torch.stack(hs, 0) * w).sum(0)
        b = self.beats_features(wav, f.shape[1]).to(f.dtype)
        x = torch.cat([f, b], dim=-1)

        x, _ = self.rnn(x)
        x = self.drop(x)
        strong = self.strong(x).squeeze(-1)
        cat = self.cat(x)
        att = self.att(x).squeeze(-1)
        if frame_mask is not None:
            m = frame_mask[:, :att.shape[1]]
            if m.shape[1] < att.shape[1]:
                m = torch.cat([m, torch.zeros(m.shape[0], att.shape[1]-m.shape[1],
                                              dtype=torch.bool, device=m.device)], 1)
            att = att.masked_fill(~m, torch.finfo(att.dtype).min)
            strong = strong.masked_fill(~m, torch.finfo(strong.dtype).min)
        w = torch.softmax(att, dim=-1)
        weak = (w * strong.clamp(min=-30)).sum(-1)
        weak_cat = (torch.softmax(self.att_cat(x), dim=1) * cat).sum(1)
        return {"strong_logit": strong, "cat_logit": cat,
                "weak_logit": weak, "weak_cat_logit": weak_cat,
                "onset_logit": self.onset(x).squeeze(-1),
                "offset_logit": self.offset(x).squeeze(-1)}

    def param_groups(self, lr_head=1e-3, lr_encoder=3e-5):
        enc, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith(("fine.encoder.", "beats.")) else head).append(p)
        return [{"params": enc, "lr": lr_encoder}, {"params": head, "lr": lr_head}]

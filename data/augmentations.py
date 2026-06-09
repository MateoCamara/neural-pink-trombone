"""On-the-fly waveform augmentation to close the synthetic -> real-voice domain gap.

The beta-VAE was trained only on clean, synthetic Pink Trombone audio, so it collapses
on real microphone input (in-domain param error ~5%, but a steady real vowel makes the
predicted articulatory params wander ~48% of their range). These augmentations perturb
ONLY the acoustics of each training clip -- never the articulatory parameter labels --
so the model learns to be invariant to the things that differ between synthetic audio and
a real mic: level, noise, room, mic colouration and band-limiting.

Apply to the raw waveform in the dataloader, BEFORE the mel spectrogram is computed and
BEFORE normalizar_mel_spec. Pure numpy/scipy -> no extra dependencies, portable to any
training machine.

Usage:
    aug = WaveformAugment(sample_rate=48000)          # defaults are sensible
    wav_aug = aug(wav_1d_float, sr=48000)             # wav_1d_float: 1-D np.float32 in ~[-1,1]
"""
import numpy as np
from scipy import signal


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


class WaveformAugment:
    """Random chain of acoustic augmentations. Each stage fires with its own probability;
    ``p`` gates whether the whole chain runs at all (so some clips stay clean)."""

    def __init__(self,
                 sample_rate=48000,
                 p=0.9,                      # prob. of applying ANY augmentation to a clip
                 gain_prob=0.8, gain_db=(-12.0, 6.0),
                 noise_prob=0.8, snr_db=(5.0, 35.0),
                 reverb_prob=0.5, rt60_s=(0.1, 0.6), reverb_wet=(0.2, 0.7),
                 tilt_prob=0.6, tilt_db=(-8.0, 8.0),
                 lowpass_prob=0.3, lowpass_hz=(4000.0, 9000.0),
                 seed=None):
        self.sr = sample_rate
        self.p = p
        self.gain_prob, self.gain_db = gain_prob, gain_db
        self.noise_prob, self.snr_db = noise_prob, snr_db
        self.reverb_prob, self.rt60_s, self.reverb_wet = reverb_prob, rt60_s, reverb_wet
        self.tilt_prob, self.tilt_db = tilt_prob, tilt_db
        self.lowpass_prob, self.lowpass_hz = lowpass_prob, lowpass_hz
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._seeded = False

    def _ensure_worker_seed(self):
        """With num_workers>0 the dataset (and this rng) is pickled to each worker, so they
        would all draw the SAME augmentation sequence. Re-seed once per worker from PyTorch's
        per-worker seed so every worker augments differently."""
        if self._seeded:
            return
        self._seeded = True
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
            if info is not None:
                self.rng = np.random.default_rng((int(info.seed) & 0xFFFFFFFF) ^ (info.id + 1))
        except Exception:
            pass

    # --- individual stages -------------------------------------------------
    def _gain(self, x):
        g_db = self.rng.uniform(*self.gain_db)
        return x * (10.0 ** (g_db / 20.0))

    def _tilt(self, x):
        """Linear-in-frequency spectral tilt of `tilt_db` total across the band.
        Mimics mic / channel frequency colouration."""
        total_db = self.rng.uniform(*self.tilt_db)
        n = len(x)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, d=1.0 / self.sr)
        nyq = self.sr / 2.0
        gains = 10.0 ** ((total_db * (f / nyq)) / 20.0)
        return np.fft.irfft(X * gains, n=n)

    def _lowpass(self, x):
        cutoff = self.rng.uniform(*self.lowpass_hz)
        cutoff = min(cutoff, 0.99 * self.sr / 2.0)
        b, a = signal.butter(4, cutoff / (self.sr / 2.0), btype='low')
        return signal.lfilter(b, a, x)

    def _reverb(self, x):
        """Convolve with a synthetic exponentially-decaying room impulse response."""
        rt60 = self.rng.uniform(*self.rt60_s)
        L = max(8, int(rt60 * self.sr))
        t = np.arange(L)
        decay = np.exp(-6.908 * t / L)               # ~ -60 dB at the tail end
        rir = self.rng.standard_normal(L) * decay
        rir[0] += 1.0                                 # direct path
        wet = signal.fftconvolve(x, rir)[:len(x)]
        wet *= _rms(x) / _rms(wet)                    # preserve loudness
        mix = self.rng.uniform(*self.reverb_wet)
        return (1.0 - mix) * x + mix * wet

    def _noise(self, x):
        snr = self.rng.uniform(*self.snr_db)
        sig_p = np.mean(x ** 2)
        if sig_p <= 0:
            return x
        noise_p = sig_p / (10.0 ** (snr / 10.0))
        noise = self.rng.standard_normal(len(x)) * np.sqrt(noise_p)
        return x + noise

    # --- chain -------------------------------------------------------------
    def __call__(self, wav, sr=None):
        self._ensure_worker_seed()
        x = np.asarray(wav, dtype=np.float64).reshape(-1)
        if x.size == 0 or not np.isfinite(x).all():
            return np.nan_to_num(x).astype(np.float32)
        if self.rng.random() >= self.p:
            return x.astype(np.float32)               # leave this clip clean

        if self.rng.random() < self.tilt_prob:
            x = self._tilt(x)
        if self.rng.random() < self.lowpass_prob:
            x = self._lowpass(x)
        if self.rng.random() < self.reverb_prob:
            x = self._reverb(x)
        if self.rng.random() < self.gain_prob:
            x = self._gain(x)
        if self.rng.random() < self.noise_prob:       # noise last: SNR vs the processed signal
            x = self._noise(x)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32)

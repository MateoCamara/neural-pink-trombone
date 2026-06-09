# Phase A: data augmentation to close the synthetic → real-voice domain gap

## Why
The β-VAE predicts Pink Trombone articulatory params from a mel spectrogram. On the
**synthetic** test set it is accurate (≈5 % mean error per param, corr ≈0.94), but on a
**real human voice** it collapses: a steady sustained vowel makes the predicted params
wander ~48 % of their range ("it can't stabilise"). Root cause = a textbook **domain gap**:
the model was trained 100 % on clean synthetic Pink Trombone audio with **zero augmentation**
and never saw real-mic acoustics (noise, room, mic colouration, level).

Phase A teaches invariance to exactly those nuisances by augmenting the training audio.

## What changed (this repo)
- **`data/augmentations.py`** (new): `WaveformAugment` — pure numpy/scipy, no new deps.
  Random gain, additive noise at random SNR, synthetic-RIR reverb, spectral tilt/EQ,
  low-pass. Applied to the raw waveform; **labels are untouched** (augmentation changes
  acoustics, not articulation). Re-seeds per DataLoader worker.
- **`data/dynamic_spectrogram_dataloader.py`**: in `__getitem__`, after `torchaudio.load`
  and before the mel, applies the augmenter **only on the `train` split**. Reads its config
  from `data_params` (`augment`, `sample_rate`, `augmentation: {...}`).
- **`configs/config_betaVAESynth_dynamic_aug.yaml`** (new): the β-VAE config + an
  `augmentation` block. Identical model/loss to `config_betaVAESynth_dynamic_1.yaml`.

It was smoke-tested locally (augmented batches flow through the model + backward OK).

## Run the full training (big machine)
Prereqs: the original training env (lightning, torch, torchvision, tensorboard, matplotlib,
librosa, scipy …) and the dataset extracted at
`../neural-pink-trombone-data/pt_dataset_dynamic_simplified/{train,test}` + the
`*_interpolated.json` files (as the original config expects).

```bash
python run.py -c configs/config_betaVAESynth_dynamic_aug.yaml
```
Checkpoints land in `logs/betaVAESynth_dynamic_aug/version_X/checkpoints/`.
Tune `data_params.num_workers` to the machine. Watch `val_loss` in TensorBoard.

## Deploy the new model back into the live demo (`C:\dev\npt-demo`)
1. Slim the new `last.ckpt` (or best) with `backend/slim_checkpoints.py` (point it at the
   new checkpoint) → overwrite `backend/models/betavae_dynamic_1.ckpt`.
2. Restart the backend and re-test.

> NOTE: the demo's inference had a **mel-normalisation bug** (used `[-40, 50]` dB vs the
> training `[-65, 45]`). Already fixed in `npt-demo/backend/voiceapp/utils.py`
> (`min_spec_value=-65, max_spec_value=45`) — keep inference and training consistent.

## How we'll measure improvement
- **In-domain**: mean per-param error vs ground truth on the synthetic test set (should stay low).
- **Real voice (the real test)**: feed the AES real vowel recordings
  (`…/resultados/neural-pink-trombone/original/*.wav`) and measure per-param **range %** on a
  steady vowel — target: far below the current ~48 %. Plus listening in the demo.

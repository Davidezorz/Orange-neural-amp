# Orange Neural Amp 🍊

Neural network modeling of a guitar amplifier / distortion pedal. The project explores several architectures for **black-box** and **gray-box amp modeling**, by learning a sample-by-sample mapping from a *clean input signal* to the corresponding distorted (amped) output — and compares them on a recorded capture of a *Distort* pedal.

The whole training pipeline lives in **`main.py`**: data loading, preprocessing, model construction, training loop (via PyTorch Lightning) and evaluation.

---

## 📂 Repository layout

| File / folder | Purpose |
|---|---|
| `main.py` | Entry point. Builds the dataset, instantiates a model, trains and evaluates. |
| `preprocessing.py` | Audio loading, delay estimation, alignment, cropping, train/val split. |
| `data.py` | `AudioDataset` — chunks long waveforms into fixed-size training samples. |
| `lightning_model.py` | `LightningModel` wrapper: losses, optimizer, LR schedule, logging. |
| `losses.py` | Loss functions (ESR, Weak-ESR, MSE, multi-resolution STFT using auraloss). |
| `models/` | Different architectures tried so far. |
| `utils.py` | Plotting helpers, parameter count, device selection. |
| `.data/` | Input captures (`T3K-sweep-v3.wav`, `v3_0_0 Sparkle Combo Distort.wav`). |lightning_logs
| `lightning_logs/` | Saved checkpoints. |
| `.weights/` | Manually saved weights. |

---

## 🧠 Models

The `models/` folder contains all the architectures that were explored for the same task. They are all compatible with the `LightningModel` wrapper and can be selected in `main.py` by uncommenting the desired line.

### `LSTM.py` — `SimpleAmpLSTM`
A small recurrent baseline. A single-layer LSTM with learnable initial `(h₀, c₀)` states maps each input sample to a hidden vector, and a linear "head" projects it back to a single output sample. Includes a burn-in window and truncated BPTT during training, and a chunked forward pass at evaluation to work around a known PyTorch MPS bug on sequences longer than 65535.

### `biquads_model.py` — `BiquadsBlock` / `BiquadsModel`
A direct port of [Wright et al., "Differentiable White-Box and Gray-Box Modelling of Vintage Audio Effects" (2021)](https://arxiv.org/pdf/2103.08709) without the conditioning input. Each block contains a learnable cascade of `K` parametric biquad filters (peaking-EQ style: `db_gain`, `f`, `Q`). The model chains `S` such blocks, separated by a `tanh` nonlinearity and a learnable per-block gain. Four filter backends are implemented and selectable per train/eval: a Python `for` loop (`seq`), FFT over the whole sequence (`fft`), short-time FFT (`sfft`), and a parallel-scan SSM implementation (`ssm`).

### `mamba2.py` — `Mamba2`
A from-scratch reimplementation of the Mamba2 selective state-space layer (chunked SSD with a 1D depthwise conv front-end, SiLU-gated RMSNorm, learnable `dt`, `A_log`, `D`, optional learnable initial state). Used as a drop-in SSM block inside `StateSpaceModels.Model`.

### `StateSpaceModels.py` — `Model`
A pure state-space architecture: an input projection, a stack of `D` LRU-style blocks (Mamba2/LRU + nonlinearity + residual), and an output projection. No biquad priors — fully learned linear recurrences.

### `s4d.py` — `ModelS4D`
A minimal, pedagogical S4D (diagonal state-space) model with complex diagonal `A`, Vandermonde-style kernel construction and a learnable per-channel `dt`.

### `model_test.py` — under active development
This file is the **current research direction**. It contains:

- `BiquadsBlock` — a single-channel biquad block equivalent to the one in `biquads_model.py`, but evaluated exclusively through the parallel-scan (SSM) backend. Useful as a clean reference.
- `SelectiveBiquadsBlock` — an experimental time-varying biquad block where a small causal 1D convolution produces per-sample deltas to the biquad parameters (an attempt at making the filters non-LTI). **Currently not working — do not use.**
- `ModelTest` — `S` cascaded single-channel biquad blocks with `tanh` nonlinearities and per-block learnable gains (the single-channel counterpart of the model below).
- `ModelTestMC` / `MultiChannelBiquad` — **the model currently in development**, described in detail below. This is the best-performing model so far.

---

## 📚 The model in development: `ModelTestMC` / `MultiChannelBiquad` 

This project implements a **multi-channel, fully differentiable cascade of parametric biquad filters**, interleaved with learned linear mixing stages and non-linearities. It bridges classical Digital Signal Processing (DSP) with modern hardware-aware State Space Models (SSMs). 

The full model instantiated in `main.py` is:

```python
ModelTestMC(S=10, K=8, C=1, H=8, sampling_rate=48000)
```

With `S=10` cascaded blocks, `K=8` filters per channel, and `H=8` parallel hidden channels, the model achieves an **ESR ≈ 0.016 on the validation set using only ~2.5K parameters**.

### Architecture Overview 

```text
x ──► in_proj (1 → H) ──► [ MultiChannelBiquad (H, K) ] ──► mid_proj + tanh(+bias) ──► ... ──► out_proj (H → 1) ──► y
                                      S = 10 blocks, K = 8 biquads/channel
```

This architecture acts as a highly efficient inductive bias for physical amp modeling. A real guitar amplifier is essentially a chain of linear filters (tone stacks, cabinet EQs) interleaved with nonlinear stages (tubes, clipping diodes). `ModelTestMC` abstracts this physical reality into a deep neural network, projecting a single audio channel into `H` dimensions, applying massive parallel parametric EQ, mixing the channels, and applying static `tanh` distortion.

---

## 📋 The Core Engine: `MultiChannelBiquad`

The heart of the network is the `MultiChannelBiquad` module. For each of the `H` channels, it contains a cascade of `K` parametric biquad filters. Instead of learning raw filter coefficients (which causes extreme instability), the network learns three physically meaningful parameters per filter:

1. **Center Frequencies ($f\_raw$):** Passed through $0.25(\sin(x) + 1)$ and cumulatively summed along the $K$ dimension. This guarantees a strictly ascending frequency order per channel, ensuring the $K$ filters seamlessly tile the frequency spectrum from 0 Hz to Nyquist without colliding.
2. **Band Gain ($db\_gain$):** The peak/cut gain in dB.
3. **Quality Factor ($Q\_raw$):** Passed through a sigmoid and bounded by a dynamic $Q_{max}$ mask. The lowest and highest filters are clamped to $Q_{max}=1$ to prevent low-frequency DC explosions or Nyquist aliasing, while mid-band filters can reach $Q_{max}=3$ for sharp resonances.

These parameters are mapped to standard RBJ Audio-EQ-Cookbook peaking biquad coefficients ($b_0, b_1, b_2, a_1, a_2$). To allow the optimizer to explore the parameter space freely, they are excluded from AdamW's weight decay.

---

## 🧮 The Math: Biquads as State Space Models

Cascading 2nd-order Infinite Impulse Response (IIR) filters using standard `for` loops is computationally devastating on GPUs and prone to exploding gradients (accumulated floating-point errors). 

To solve this, `MultiChannelBiquad` dynamically translates the 2nd-order biquad into a **1st-order complex State Space Model**, which is then computed in parallel across the time domain in $O(L)$ time using a parallel scan (`pscan`).

### Proof and Derivation (Partial Fraction Decomposition)

The standard transfer function of a digital biquad is:

$$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$

Our goal is to split this 2nd-order system into a direct volume gain and two independent, parallel 1st-order complex filters. 

**Step 1: Extract the Direct Gain ($b_0$)**
By subtracting $b_0$ from $H(z)$ via polynomial long division, we isolate the remaining fraction:

$$H(z) = b_0 + \frac{(b_1 - b_0 a_1)z^{-1} + (b_2 - b_0 a_2)z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$

Let $p_1 = b_1 - b_0 a_1$ and $p_2 = b_2 - b_0 a_2$. The remaining transfer function is now:

$$H_{rem}(z) = \frac{p_1 z^{-1} + p_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$

**Step 2: Factor the Denominator (The Poles)**
We find the roots (eigenvalues/poles) of the denominator by solving the characteristic equation $z^2 + a_1 z + a_2 = 0$.
The discriminant is $D = a_1^2 - 4a_2$. The complex poles are:

$$\lambda_1 = \frac{-a_1 + \sqrt{D}}{2}, \quad \lambda_2 = \frac{-a_1 - \sqrt{D}}{2}$$

Notice that subtracting the poles yields exactly the square root of the discriminant: $\lambda_1 - \lambda_2 = \sqrt{D}$.

**Step 3: Heaviside Cover-Up Method (The Residues)**
We factor the denominator and split the fraction into two parallel 1st-order systems with unknown complex weights (Residues $R_1$ and $R_2$):

$$\frac{p_1 z^{-1} + p_2 z^{-2}}{(1 - \lambda_1 z^{-1})(1 - \lambda_2 z^{-1})} = \frac{R_1 z^{-1}}{1 - \lambda_1 z^{-1}} + \frac{R_2 z^{-1}}{1 - \lambda_2 z^{-1}}$$

To solve for $R_1$, we multiply both sides by $(1 - \lambda_1 z^{-1})$ and evaluate at $z^{-1} = \frac{1}{\lambda_1}$ to eliminate $R_2$:

$$R_1 = \frac{p_1 \lambda_1 + p_2}{\lambda_1 - \lambda_2} = \frac{p_1 \lambda_1 + p_2}{\sqrt{D}}$$

By symmetry, evaluating at $z^{-1} = \frac{1}{\lambda_2}$ gives:

$$R_2 = \frac{p_1 \lambda_2 + p_2}{-\sqrt{D}}$$

**Step 4: Time-Domain Reconstruction**
The Z-transform fraction $\frac{1}{1 - \lambda z^{-1}}$ maps perfectly to the 1st-order linear recurrence $h[n] = \lambda h[n-1] + x[n]$. 
The $z^{-1}$ in the numerator simply delays this sequence by one sample ($h[n-1]$). 

Therefore, the final mathematically identical time-domain output of the biquad is:

$$y[n] = b_0 x[n] + R_1 h_1[n-1] + R_2 h_2[n-1]$$

Where $h_1$ and $h_2$ are computed simultaneously using hardware-aware parallel scans. This bypasses the sequential bottleneck and STFT overlap-add artifacts entirely.

---

## 📝 Note on `pscan` Complex Number Support 

To execute the math above natively in `torch.cfloat`, a minor modification was made to the `mambapy.pscan` backward pass. PyTorch requires complex conjugates for gradient accumulation in complex vector spaces. Two lines were added to `models/pscan.py` to ensure proper gradient flow:

```python
        A_in = A_in.conj()
        X = X.conj()
```
This enables the scan to natively handle the highly resonant, complex conjugate eigenvalues generated by the parametric EQ without crashing during backpropagation.
---

## 🔧 Preprocessing 

Defined in `preprocessing.py`. The pipeline is driven by a `DataInfo` dataclass that encodes where the input capture has its known landmarks:

```
(0:00-0:09)    Validation 1
(0:09-0:10)    Silence
(0:10-0:12)    Two "blips" — short reference impulses at known sample indices
(0:12-0:15)    Chirps
(0:15-0:17)    Noise
(0:17-3:00.5)  General training data
(3:00.5-3:01)  Silence
(3:01-3:10)    Validation 2
```

The `Preprocessing` class then performs, in order:

1. **Resample** both files to the target `rate` (48 kHz here) and downmix to mono.
2. **Delay estimation (synchronization) using the blips.** A window is cut around each known blip location, averaged across both blips, and compared against the background-noise level. The first sample that exceeds an absolute+relative threshold gives the offset between input and output — this is how the two recordings are time-aligned. The "lookback / lookahead" window is configurable (5 000 samples lookahead in `main.py`).
3. **Gain compensation** (optional, in dB).
4. **Trimming** to a common length.
5. **Validation-section consistency check** — the two validation segments are compared with ESR; if they don't match, the alignment is considered unreliable and the user is warned.
6. **Train / validation split** — the long middle section becomes the training data, the longest of the two validation segments becomes the held-out validation set.
7. **Normalization** — a single scalar `norm_factor = 1 / max(|y_output|)` is applied to the output target so the model trains on signals in `[-1, 1]`.

After this, `data.AudioDataset` chops the train waveforms into overlapping `chunk_size`-sample windows (default 2¹⁴ = 16384 samples, with `chunk_size // 2` stride) and the validation set is passed as a single contiguous chunk.

---

## 📉 Losses 

All losses are implemented in `losses.py` and combined inside `lightning_model.py`.

The current training objective is:

```
loss = 0.1 · WeakESR + 0.9 · ESR
```

with a 512-sample `warmup` window at the start of every chunk that is excluded from the loss (to avoid penalizing the model's transient state).

### Available losses

| Loss | Notes |
|---|---|
| `MSELoss` | Standard mean-squared error. Exposed for completeness. |
| `ESRLoss` | **Error-to-Signal Ratio**: `MSE(pred, target) / mean(target²)`. This is the metric reported as `val_esr` and used for checkpoint selection. |
| `WeakESRLoss` | Same numerator, but the denominator is `coef + (1 − coef) · mean(target²)` with `coef = 0.1`. Prevents the loss from being dominated by silent/low-energy regions. |
| `MultiResolutionSTFTLoss` | Wraps `auraloss.freq.STFTLoss` averaged over multiple FFT / hop / window sizes (default `[1024, 2048, 512]`). Operates in the spectral domain — useful for time-frequency-aware training, but currently computed and logged rather than added to the objective. |
| `PreEmphasisFilter` | A simple `y[n] − 0.1·y[n−1]` high-pass filter exposed as a `nn.Module` (handy as a differentiable pre-emphasis layer, not currently used as a loss). |

---

## ▶️ Running it 

```bash
pip install -r requirements.txt
python main.py
```

`main.py` will:

1. Load `.data/T3K-sweep-v3.wav` (clean input) and `.data/v3_0_0 Sparkle Combo Distort.wav` (amped output).
2. Run the preprocessing pipeline (delay estimation, alignment, train/val split, normalization).
3. Build the model selected at the top of `__main__` — by default `ModelTestMC(S=10, K=8, C=1, H=8, sampling_rate=48000)`.
4. Wrap it in `LightningModel` and train for `max_epochs` (default 12) with `chunk_size = 2¹⁴`, `batch_size = 8`, and AdamW + a linear LR decay.
5. Checkpoint the 3 best models (lowest `val_esr`) to `.weights/`.
6. After training, render the model's prediction on the validation segment to `.data/last_run.wav`, plot the input / target / predicted waveforms, and — for biquad-based models — print the learned per-block biquad coefficients.

---
🍊 :)
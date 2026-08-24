# Visual fUS SBIND adaptation audit

Status: **SBIND-adapted classification baseline**, not a full reproduction of SBIND.

Audited sources:

- Hosseini & Shanechi, ICML 2025 paper: <https://proceedings.mlr.press/v267/hosseini25a.html>
- Official repository: <https://github.com/ShanechiLab/SBIND>
- Official main model: `sbind/sbind.py`
- Official recurrent cells: `sbind/models/rnn_cells.py`
- Official convolution and attention components: `sbind/models/models.py`
- Official behavior/neural model configuration: `sbind/models/model_helpers.py`
- Official trainer: `sbind/sbind_trainer.py`
- Official runnable configuration: `tutorial.py` and `tutorial.ipynb`

The repository tree audited at `c00e006dd298cb8c51d1543641fb3be77cc585ae` does not contain
`sbind/convsbind.py`, although the README still names that path and the class `CONVSBIND`.
The actual checked-in main implementation is `sbind/sbind.py`, whose public class is `SBIND`.
This audit therefore treats `sbind/sbind.py` as the authoritative replacement for the stale
README path; no official source was copied into this project.

## 1. Original SBIND input form

SBIND takes raw neural-image time series, represented by the implementation as
`[batch, sequence, channel, height, width]`. The paper defines each observation as
`Y_k in R^(n_y x H x W)` and uses one neural-image channel in all reported experiments.
The paper fUS experiment uses 30 frames per 15-second trial, acquired at 2 Hz, with images
cropped from 128x132 to 128x128. Behavior is intermittently available at only three samples
before reward. The official sequence data loader supplies neural frames, behavior, and an
availability mask.

The present project is materially different: each sample is one audited visual-stimulus block,
not a complete behavioral trial. Its input is the existing clean4 tensor `[4,128,501]`, and the
four frames retain their stored chronological order.

## 2. Encoder K

The paper and official configuration use three 5x5 convolutional layers with strides
`[2,2,1]`. The first two layers have 32 channels and the final layer maps to the relevant latent
channel count. Batch normalization and LeakyReLU follow convolutional layers. The tutorial has
a residual connection on its second encoder layer. The encoder processes each time point
spatially and maps a 128x128 image to a latent map of 32x32.

V1 retains those layer counts, kernels, strides, channels, normalization, activation, and the
official tutorial's second-layer residual. Our images are 128x501, so symmetric stride-2 layers
produce approximately 32x126 rather than 32x32. The only spatial-size adaptation is a fixed
`AdaptiveAvgPool2d((32,32))` after the third encoder layer. This was chosen before performance
testing and is identical for every session, fold, seed, and model. It is a necessary geometry
adapter, not a tuned feature module.

## 3. ConvRNN dynamics

The behavior-relevant branch in the paper has eight latent channels (`n1=8`) and a recurrent
transition with one 3x3 convolution, stride 1. Equation A.1 updates the state as
`X_(k+1) = f_A(X_k) + K(Y_k)`. The official `ConvRNNCell` implements the same additive form
when the encoder output and state have equal channels; its alternate unified form concatenates
state and encoded input before convolution.

V1 uses the additive Equation A.1 form, eight 32x32 state channels, a single 3x3 local
transition, and four recurrent updates in the original clean4 order. The state is reset to zero
for every block sample. Classification uses the final updated state after all four frames.

## 4. Self-attention

Official SBIND applies patch self-attention inside `f_A`, after the local convolution. The paper
uses 8 heads, 256-dimensional Q/K/V embeddings, learnable positional information, and reports a
frozen patch size of 8 in Table A.1 (16 patches for a 32x32 latent map). Appendix A.1.2 describes
optional 1x1 channel reduction to `c=2`, patch flattening, layer normalization before and after
attention, a 1x1 channel restoration, and a learned scaled residual initialized at 0.15. The
official tutorial uses `learnable_1d` positions and exposes `att_reduced_ch`.

`SBIND-adapted` retains patch size 8, 8 heads, embedding dimension 256, reduction to 2 channels,
learnable 1D positional values, both layer norms, channel restoration, and the scaled residual.
The official attention dropout default of 0 is retained.
`SBIND-adapted-NoAtt` replaces only this global-attention operation with identity; its encoder,
ConvRNN, decoder, data, folds, seeds, and optimizer are otherwise identical.

Compatibility note: current official `MHAScaledDotProduct` calls
`torch.nn.functional.scaled_dot_product_attention`, an API introduced in PyTorch 2.0 and absent
from the frozen server PyTorch 1.12.1. V1 does not alter the environment. It evaluates the same
scaled-dot-product equation using `matmul`, division by `sqrt(head_dim)`, `softmax`, dropout, and
the same QKV/output projections. These operations all exist in PyTorch 1.12.1.

## 5. Behavior decoder D

The paper's decoder D maps the behavior-relevant state to behavior. For WFCI it downsamples
32x32 to 4x4 using three stride-2 convolutional layers, then uses a fully connected mapping. The
reported fUS configuration uses four 5x5 stride-2 convolutional layers, 16 channels, channel
dropout, and a 16-unit fully connected representation. Categorical behavior uses cross-entropy;
the original fUS task has binary or factorized multi-direction outputs depending on session.

V1 uses the official fUS decoder geometry but fixes the output to two logits. The four
convolutions use 16 channels, 5x5 kernels, stride 2, batch normalization, LeakyReLU, and the
benchmark-fixed dropout 0.25, followed by a 16-unit hidden layer and a two-class linear layer.
The objective is ordinary cross-entropy. No additional head or auxiliary loss is present.

## 6. Original neural prediction target

Full SBIND predicts the next neural image with convolutional neural decoders `C(1)` and `C(2)`.
The paper's neural objective combines L2, L1, and gradient-difference losses. Official code
also supports Gaussian/image-Gaussian and Poisson observation variants and additional
step-ahead predictions. The trainer uses distinct optimizers for the current training phase.

V1 deletes all neural-image decoders and all next-frame neural prediction targets. It does not
use MSE, L1, gradient loss, Poisson likelihood, reconstruction, or future-frame labels.

## 7. Original two-phase behavior-relevant/residual training

Full SBIND first fits ConvRNN1, K(1), and D(1) for behavior decoding; it then freezes that path
while fitting C(1) for neural prediction. Phase 2 fits ConvRNN2, K(2), and C(2) to residual
neural dynamics after the behavior-relevant prediction. The full state concatenates `X(1)` and
`X(2)`. The official trainer mirrors this separation with dedicated optimizer instances.

V1 keeps only the supervised behavior-relevant main branch. It has no ConvRNN2, residual-state
channels, neural residual, sequential optimization phases, or disentanglement claim. A single
optimizer fits encoder, recurrence, attention (Full only), and classification decoder jointly.

## 8. Original paper fUS experiment setting

The paper uses 13 non-human-primate fUS sessions from a memory-guided saccade/reach task, with
2- or 8-direction targets. Each 15-second trial has 30 frames at 2 Hz and sessions have
154.77 +/- 93.75 successful trials. Images are high-pass filtered, rolling-window voxel-wise
z-scored, spatially pillbox filtered, and cropped to 128x128. Behavior loss is masked to three
target-fixation samples before reward; inference uses the last trial state. Evaluation reports
10-fold session-level results. Table A.1 reports fUS batch size 30, sequence length 30, AdamW-like
training at learning rate 1e-3, StepLR, 80 epochs, and weight decay 1e-6. The architecture uses
`n1=8`, `n2=24`, 32x32 latent maps, 3-layer K, 3x3 recurrence, 8 heads, patch size 8, embedding
dimension 256, and learnable positions.

The checked-in `SBINDTrainer` constructor itself defaults to AdamW with learning rate 1e-3 and
weight decay 1e-8; this differs from the paper table's 1e-6. Neither value is imported into V1:
the current clean4 benchmark's predeclared weight decay 1e-3 is used for fair comparison.

This project does not reuse the paper's task labels, preprocessing, trial window, behavior mask,
fold construction, residual branch, batch size, schedule, epochs, or neural loss. Paper results
cannot be numerically compared with this baseline.

## 9. Why the present visual classification task cannot reproduce SBIND verbatim

Our target is stimulus presence within one visual block:
`grating + dot` versus `stop_after_grating + static`. A complete cycle has four blocks, and every
block is a separate sample. There are only four clean center frames per sample, the image width
is 501, and there is no continuous trial-level behavior stream or next-frame neural target.
Consequently, a 30-frame recurrent sequence, intermittent behavior mask, one-step neural image
prediction, behavior/relevant-residual disentanglement, and original fUS directional decoder
would change the scientific question or require unavailable targets. Calling this a full SBIND
reproduction would therefore be incorrect.

The existing project path is reused rather than reimplemented:

- Loader: `load_block_sequence_session`, which asserts `[N,4,128,501]`, finite values, four
  complete ordered blocks per cycle, and two stimulus/two non-stimulus labels per cycle.
- Clean4 indices: read directly from `/clean4/X` and `/clean4/original_frame_indices`; this code
  never recalculates positions. Session 710, for example, loads 18 cycles/72 samples and the
  first cycle indices `[3,4,5,6]`, `[10,11,12,13]`, `[18,19,20,21]`, `[25,26,27,28]`.
- Folds: `grouped_cv_splits(..., max_folds=10)` and byte-equivalent tabular comparison to each
  formal `block_clean4_binary_v1/session_*/split_manifest.csv`; any mismatch stops execution.
- Normalization: existing train-fold-only arcsinh and pixel-wise z-score implementation.
- Training loop: existing fixed-epoch cross-entropy loop, extended only with an optional
  DataLoader worker count whose default remains zero.

## 10. Exact V1 keep/delete protocol

Kept from SBIND's architectural idea:

- per-frame shallow spatial convolution encoder K;
- an image-shaped eight-channel recurrent state at 32x32;
- one local 3x3 convolutional transition over four chronological frames;
- patch self-attention inside the recurrent transition for Full SBIND;
- the controlled NoAtt removal;
- a convolutional behavior/classification decoder;
- categorical cross-entropy.

Deleted or adapted:

- deleted ConvRNN2, neural decoder C, future-image prediction, neural losses, residual dynamics,
  two-phase training, stateful cross-sample recurrence, and paper-specific behavior masking;
- adapted 128x501 encoder output to 32x32 by one fixed pooling layer;
- adapted D to a two-logit block classifier;
- adapted fused attention to an equation-equivalent PyTorch 1.12 implementation;
- used the already frozen benchmark settings for fairness: AdamW, learning rate 1e-3, weight
  decay 1e-3, batch size 16, dropout 0.25, 40 fixed epochs, seeds 0/1/2, no early stopping;
- used identical clean4 samples and cycle folds for NoAtt, Full, and historical comparisons.

No architecture or hyperparameter may be changed after inspecting test performance. Batch size
may be reduced uniformly only for CUDA OOM, using a new output directory/config fingerprint;
all other settings remain fixed. Formal outputs belong only to server runs and are kept under
`outputs/sbind_visual_binary_v1/`.

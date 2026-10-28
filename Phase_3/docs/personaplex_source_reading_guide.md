# PersonaPlex upstream source — reading guide

This guide follows the checked-out NVIDIA source without running it. The
upstream checkout currently points to commit
`3428dfd95309a7f3c84fd93259ded0f810d1ff91` (`Update README.md`). Line numbers
below refer to that revision. Inference and future experiments are intended to
run on the remote GPU platform; this guide is for source reading on the local
machine.

## Start with the two inference entry points

PersonaPlex has two useful entry points. Read the WAV path first because it is
linear and easier to follow; then read the WebSocket path to see how live audio
is buffered and streamed.

| Entry point | Read | What it does |
|---|---|---|
| WAV/offline | [`moshi/offline.py`](../upstream/personaplex/moshi/moshi/offline.py), lines 153–318 (`run_inference`) and 321–430 (`main`) | Loads models, primes prompts, replays input audio frame by frame, writes output WAV and text JSON |
| Live/realtime | [`moshi/server.py`](../upstream/personaplex/moshi/moshi/server.py), lines 357–480 (`main`) and 482 (`main()` call) | Loads models once, starts the aiohttp WebSocket server and serves the UI |

`server.py` calls `main()` at module level under `torch.no_grad()` at line 482;
it is intended to be launched as `python -m moshi.server`. `offline.py` uses a
normal `if __name__ == "__main__"` guard at lines 430 onward.

## Realtime call chain, in order

### 1. CLI and model construction

Read [`server.py`](../upstream/personaplex/moshi/moshi/server.py):

1. Lines 357–426: define CLI arguments, resolve voice prompts and UI assets,
   choose device, and seed startup.
2. Lines 427–447: download or resolve `config.json`, Mimi checkpoint, tokenizer,
   and Moshi checkpoint; call the loaders.
3. Lines 448–460: create `ServerState`, warm up, then register `/api/chat`.
4. Lines 461–479: optionally serve the static browser UI and start aiohttp.

Then read [`models/loaders.py`](../upstream/personaplex/moshi/moshi/models/loaders.py):

- Lines 39–45 name the 24 kHz sample rate, 12.5 Hz model frame rate, checkpoint
  files and default Hugging Face repo.
- Lines 48–121 define the Mimi, Temporal Transformer, and LM/Depformer
  configuration. The main LM is configured with width 4096 and 32 layers;
  the Depformer has width 1024 and 6 layers.
- Lines 129–163 construct Mimi and set its active quantizer to 8 codebooks.
- Lines 166–260 construct/load the Moshi LM. In this revision the loader sets
  `dep_q = 16`, then handles checkpoint compatibility/weight loading. Read
  those lines rather than assuming every field in `_lm_kwargs` remains the
  runtime value.

### 2. Session and persona prefix

Read `server.py` lines 89–132 for `ServerState`: it retains two Mimi instances,
the text tokenizer, one shared `LMGen`, a lock, and the audio frame size. The
frame size is `sample_rate / frame_rate` (24,000 / 12.5 = 1,920 samples).

Read `server.py` lines 135–171 to see how `/api/chat` picks the voice prompt,
loads voice audio or precomputed embeddings, wraps the role text in `<system>`
tags, and gets the optional seed from the query.

Then read [`models/lm.py`](../upstream/personaplex/moshi/moshi/models/lm.py):

- Lines 960–984 load raw voice audio or cached voice embeddings.
- Lines 985–1063 encode each voice frame and feed it through `LMGen` while the
  other speaker's audio slots receive a synthetic sine-frame token.
- Lines 1074–1127 add silence frames around the role-text prompt; the role text
  is fed as text tokens while the audio streams receive their prompt/silence
  tokens.
- Lines 1117–1127 show the full prompt order: voice → silence → role text →
  silence. The prompt is prefilling of the existing model streams, not a
  separate model.

The browser constructs `/api/chat` and passes `text_prompt` and `voice_prompt`
in [`client/src/pages/Conversation/Conversation.tsx`](../upstream/personaplex/client/src/pages/Conversation/Conversation.tsx),
lines 56–74. For the byte protocol, read
[`client/src/protocol/encoder.ts`](../upstream/personaplex/client/src/protocol/encoder.ts),
lines 9–32: `0x00` handshake, `0x01` audio, `0x02` text, with other control
messages defined there too.

### 3. Live audio loop

Return to `server.py` and follow these ranges:

1. Lines 173–202 (`recv_loop`): receive binary WebSocket messages; message kind
   `1` is Opus audio, appended to the stream reader.
2. Lines 204–243 (`opus_loop`): decode Opus to PCM, buffer until one 1,920-sample
   frame is available, run Mimi encoding, then call `lm_gen.step(...)` once for
   each encoded frame.
3. Lines 230–242: generated frame layout is text at index 0 and audio at the
   following indices. The first eight generated audio codebooks
   (`tokens[:, 1:9]`) are decoded to outgoing speech; a non-padding text token
   is sent with the `0x02` prefix.
4. Lines 244–251 (`send_loop`): flush encoded Opus audio to the WebSocket with
   the `0x01` prefix.
5. Lines 258–305: acquire the shared model lock, seed/reset session streaming
   state, run the system prompt, send the handshake, start the receive/audio/send
   tasks, and cancel remaining tasks on disconnect. Since the lock surrounds
   the live session loop and `ServerState` owns one `LMGen`, this server
   serializes sessions that share that state.

Note the two Mimi objects: both are encoded/decoded in this code path, although
only the first decoder's PCM is sent to the client. Trace their streaming states
before changing or removing either instance.

## Offline WAV call chain

Read [`offline.py`](../upstream/personaplex/moshi/moshi/offline.py) in this order:

1. Lines 321–384: CLI inputs, output files, role/voice prompts, sampling, seed,
   device, and optional CPU offload.
2. Lines 153–224: `run_inference` loads Mimi twice, tokenizer, LM and `LMGen`;
   unlike the server, it exposes temperature/top-k and greedy decoding.
3. Lines 225–255: enter streaming mode, warm up, load voice and role prompts,
   reset states, and prefill the system prompt.
4. Lines 257–318: load the user WAV, encode it to Mimi frames, feed one frame at
   a time through `LMGen.step`, decode output audio, collect text tokens, and
   write WAV plus JSON.

For a controlled baseline replay, this is the more convenient entry point to
wrap on the remote GPU: it accepts a fixed WAV and seed and produces named
artifacts. It is still inference code; it does not implement scenario scoring
or RL rollouts.

## The model behind `LMGen.step`

Read the model files after understanding the call sites above.

### Mimi audio codec

In [`models/compression.py`](../upstream/personaplex/moshi/moshi/models/compression.py):

- Lines 102–163: `MimiModel`, its streaming state and audio metadata.
- Lines 344–380: waveform → encoder/encoder transformer → resample to model
  frame rate → vector-quantizer codes.
- Lines 398–421: codebooks → quantizer/decoder transformer → waveform.

The model operates at 24 kHz and 12.5 frames/second. Mimi is the audio codec;
it does not make the turn-taking decision.

### LMModel: Temporal Transformer and Dependent Transformer

In [`models/lm.py`](../upstream/personaplex/moshi/moshi/models/lm.py):

- Lines 218–355: `LMModel.__init__` creates per-codebook embeddings, the main
  causal Temporal Transformer and the per-codebook Depformer projections and
  output heads.
- Lines 398–455: codebook IDs are embedded and summed; the main transformer
  emits contextual states and text logits.
- Lines 457–493: `forward_depformer` predicts one audio codebook conditioned on
  the temporal state and previous text/audio token.
- Lines 495–529: batched Depformer forward used for sequence training.
- Lines 531–552: `forward_train` shifts/delays the joint sequence, calculates
  text and audio logits, and returns masks for valid targets.

Important distinction: the model has one text stream plus 16 audio codebook
streams in this checkpoint setup: 8 for agent audio and 8 for user audio. Mimi
uses 8 active codebooks per speaker. `LMGen` receives the user and agent groups
separately and marks supplied tokens so the Depformer does not sample over the
user-provided group. Follow `loaders.py:92–121,166–260`, `lm.py:739–779`, and
`lm.py:1129–1177` together to understand the channel ordering and one-frame
delays.

### LMGen: streaming state, delays and sampling

Continue in `models/lm.py`:

- Lines 556–568: generation state (`cache`, `provided`, `initial`, offset).
- Lines 646–724: `LMGen` configuration and per-session streaming cache. CUDA
  graphs wrap main-LM and Depformer calls where available.
- Lines 727–812: `prepare_step_input` places user audio, agent/prompt audio and
  text tokens into the delayed cache and selects current input/target positions.
- Lines 814–850: `step` runs the main transformer and delegates token selection.
- Lines 875–958: sample text, generate audio through the Depformer, preserve
  provided tokens, and return outputs after the configured delay.
- Lines 1129–1177: `depformer_step` autoregressively predicts the codebooks for
  one frame, feeding each selected code into the next codebook step.

To understand cache reset/propagation, read
[`modules/streaming.py`](../upstream/personaplex/moshi/moshi/modules/streaming.py),
lines 262–355. For attention's causal KV cache, read
[`modules/transformer.py`](../upstream/personaplex/moshi/moshi/modules/transformer.py),
lines 232–299 and 317–447; for the stack of transformer layers, read lines
458–720.

## What this means for P1–P3 and RL

- **P1 (in-turn adaptation):** the live user audio enters as the user-side
  Mimi codes in `server.py:224–228` / `offline.py:267–283`, then is placed into
  the joint delayed cache by `LMGen.prepare_step_input`. Generated text and
  agent audio are sampled downstream. This is where to trace whether a user
  correction arrives before the model's next continuation and what context is
  available at that point.
- **P2 (proactive intervention):** the model must choose text/audio tokens while
  user audio continues. The current server loop invokes `step` on each input
  frame; it has no separate content-trigger policy or reward verifier.
- **P3 (instruction-conditioned turn control):** role instructions enter as
  text prompt tokens during prefilling. Read `step_system_prompts` and then the
  same `LMGen` cache/step path to see how the instruction is converted into
  persistent context; there is no standalone action head in the upstream code.
- **Autograd/RL seam:** `LMGen.prepare_step_input`, `step`, and
  `process_transformer_output` are under `torch.no_grad()`; the server also
  wraps its entry point in `torch.no_grad()`. `LMModel.forward_train` is a
  differentiable sequence forward, but it is teacher-forced over token codes,
  not a ready-made online GRPO trainer. `return_logits` in `LMGen` can expose
  sampled-step logits, but that path is still inference/no-grad. A LoRA-GRPO
  implementation must explicitly connect rollout tokens/rewards to a
  gradient-bearing scoring/update path and preserve the delay/channel masks.
- **Likely LoRA target:** inspect `lm.transformer` (the main temporal model)
  separately from `lm.depformer` (within-frame audio-code generation). The
  repository does not include a LoRA trainer or adapter implementation.

For the project-level diagram and experiment module layout, see
[`architecture-and-code-reading.md`](architecture-and-code-reading.md) and
[`experiment-design.md`](experiment-design.md).

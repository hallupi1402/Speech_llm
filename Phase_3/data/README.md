# Data and manifests

Store scenario manifests and references to audio here. Prefer paths or dataset
identifiers over copying large or private audio into Git. Record the source,
license or permitted-use terms, sample rate, channel count, duration, and source
of event timestamps for every audio item.

## P03 pilot

`p03/scenarios.jsonl` contains five **draft templates**. They are not test cases
until audio and event labels have been prepared. Each pair uses the same user
recording for the `listen` and `backchannel` conditions, so the instruction is
the only planned difference.

For every pair:

1. Save mono, 24 kHz, PCM16 WAV audio at the path in `audio.path` (for example,
   `Phase_3/data/p03/audio/p03_001.wav`).
2. Replace `pending` source and license fields with the actual source and terms.
3. Set `duration_s` to the WAV duration.
4. Add `user_holds_floor` intervals for speech where the user still has the
   floor and `backchannel_opportunity` intervals where a short acknowledgement
   would fit. Times are seconds from the start of the WAV. The model does not
   receive these event timestamps; the scorer uses them.
5. Change `status` to `ready` only after audio and annotations are checked.

Keep this pilot separate from later held-out evaluation data. The five
templates contain no audio or verified ground truth by themselves.

## Downloaded benchmark: DSB-IFEval

The benchmark snapshot is stored locally at
`data/external/DSB-IFEval/` and excluded from Git. Source:
[`puneetUMD/DSB-IFEval`](https://huggingface.co/datasets/puneetUMD/DSB-IFEval).
The downloaded package contains `data/manifest.jsonl`, `data/cases.jsonl`,
`data/scripts.jsonl`, role/probe/instruction specs, and the audio files. Its
manifest has 1,038 cases; audio is organized under
`audio/<audio_key>/turn1.wav` and `turn2.wav` as mono 24 kHz WAV, with segment
timelines in `meta.json`.

The dataset card specifies **CC-BY-NC-4.0**. Use it only under that license's
terms and retain attribution. The dataset is research audio; it includes
synthetic speech and safety-sensitive scenarios. The dataset card and its
`README.md` are the source of truth for its scope, licensing, and limitations.

`p03/dsb_pilot_selection.jsonl` lists 10 selected audio situations and their
20 L1/L2 cases. It stores the original `system_instruction`, trigger window,
verifier, and paths into the downloaded dataset. The audio remains in
`data/external/DSB-IFEval/`; this selection is metadata for review and is not
yet a runnable Phase 3 scenario manifest.

For easier inspection, `p03/dsb_pilot_subset/` contains a copy of the 20 WAVs
used by those 10 situations, a `manifest.jsonl` whose audio paths point inside
that folder, and a linked index in `README.md`. The WAVs are excluded from Git.

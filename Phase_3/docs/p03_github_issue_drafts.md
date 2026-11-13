# P03 GitHub Project Issue Drafts

These four task cards are intended to be created directly inside the GitHub Project at `https://github.com/users/hallupi1402/projects/1`. Each card uses one fixed English format. Convert each draft card into a repository issue when you want it linked to `hallupi1402/Speech_llm`.

## Shared issue format

Every issue uses these headings, in this order:

1. Goal
2. Background
3. Tasks
4. Acceptance Criteria
5. Deliverables
6. Dependencies

Use the issue number prefix in each title to keep the work in order. Keep the original PersonaPlex checkout unchanged during baseline experiments.

---

## Issue 1

**Title:** `[P03-01] Create paired instruction scenarios`

### Goal
Create a small, controlled set of audio scenarios for checking whether PersonaPlex changes its turn-taking behavior when the instruction changes.

### Background
P03 concerns following instructions about when to speak, listen, give a short backchannel, or interrupt. The first experiment should use the same user audio with different instructions. For this first pass, provide the instruction when the session starts; runtime instruction injection is outside this issue.

### Tasks
- Choose two contrasting behaviors for the pilot, such as `listen` and `backchannel`.
- Create 5 initial scenarios with the same audio reused across the paired instructions.
- Record the audio reference, instruction text, relevant event time, expected behavior, and negative-control notes.
- Keep audio files outside Git when their size or license makes that appropriate; track them by path or URL in the manifest.
- Check that every pair differs in the instruction while keeping its audio and event labels the same.

### Acceptance Criteria
- There are at least 5 valid paired scenarios.
- Every scenario has an expected action and a relevant timestamp or an explicit `not_applicable` value.
- A validator catches missing fields and mismatched audio within a pair.
- The pilot does not use evaluation cases as training examples.

### Deliverables
- `Phase_3/data/p03/scenarios.jsonl`
- A short README describing fields, audio requirements, and how to add a scenario.

### Dependencies
- None.

---

## Issue 2

**Title:** `[P03-02] Collect PersonaPlex baseline rollouts on Modal`

### Goal
Run the original PersonaPlex model on the paired P03 scenarios and save its generated speech and run details.

### Background
The Modal adapter starts the upstream WebSocket server. The current server accepts `text_prompt` when a session opens. This issue tests session-start instructions only and must not change upstream model code.

### Tasks
- Add a small scenario runner under `Phase_3/src/phase3/rollout/` that connects to the existing Modal WebSocket endpoint.
- For each scenario, start a separate session for each paired instruction and send the same user audio.
- Save the generated audio and any returned text.
- Save scenario ID, instruction, model/source revision, voice prompt, sampling settings, seed, and run time alongside each output.
- Start with the 5 paired scenarios from P03-01; do not launch a large batch.

### Acceptance Criteria
- The runner completes both instruction variants for every pilot scenario.
- Outputs can be matched back to their scenario and instruction without guesswork.
- A failed or disconnected session is recorded as a failed run, not silently omitted.
- No model or upstream source changes are required for this baseline.

### Deliverables
- Modal rollout runner and usage instructions.
- Output manifest and generated audio references under `Phase_3/runs/p03/baseline/`.

### Dependencies
- P03-01.

---

## Issue 3

**Title:** `[P03-03] Measure instruction adherence and check the scoring method`

### Goal
Determine whether the original model follows the paired instructions and whether the proposed scoring correctly distinguishes good and bad behavior.

### Background
A score is useful only if it agrees with what a person hears. The first version should use simple speech timing checks plus a small manual review, rather than relying on a complex automatic judge.

### Tasks
- Measure whether the model speaks during a `listen` instruction.
- For `backchannel`, measure whether a short response occurs near the labeled opportunity.
- Compare the two instruction variants for each identical audio scenario.
- Manually review a sample of outputs and note scorer errors.
- Report failed sessions, unclear cases, and cases where the instruction had no measurable effect.

### Acceptance Criteria
- Each output has per-scenario scores and evidence, such as speech intervals and the matched event time.
- The report separates instruction adherence, timing, and false speech events.
- The scoring method has been checked against a manually reviewed sample.
- A go/no-go decision says whether the baseline failure is clear enough and the scorer reliable enough to justify a training pilot.

### Deliverables
- `Phase_3/src/phase3/evaluation/` scoring code and a short manual-review guide.
- Baseline analysis under `Phase_3/runs/p03/baseline/`.
- A written go/no-go decision.

### Dependencies
- P03-01 and P03-02.

---

## Issue 4

**Title:** `[P03-04] Run a small P03 improvement pilot`

### Goal
If the baseline shows a repeatable instruction-following failure and the scorer is reliable, test whether a small, low-rank model update improves it.

### Background
This is a feasibility pilot, not full-scale training. Keep the original PersonaPlex checkpoint as the reference. Do not add mid-session instruction injection in this pilot; that requires a separate change to the server and streaming model input.

### Tasks
- First confirm the P03-03 go decision and profile one complete training step on the available Modal GPU.
- Use short scenarios and a small number of alternative outputs per scenario.
- Freeze the audio encoder/decoder and keep the model update limited to a small set of language-model parameters.
- Compare the updated model with the original on the same held-out paired scenarios.
- Record GPU memory, elapsed time, score changes, false actions, and speech-quality regressions.
- Stop if a complete training step does not fit or the reward does not distinguish the candidate outputs.

### Acceptance Criteria
- The pilot completes at least one real update step, or documents the specific measured blocker.
- Evaluation uses scenarios not used to adjust the scoring method or perform training.
- Results include original-versus-updated comparisons and the individual score components.
- The report states whether to continue, revise, or stop; a higher training score alone is not considered success.

### Deliverables
- Small training configuration and Modal entrypoint under `Phase_3/src/phase3/training/`.
- Adapter/checkpoint reference and run metadata under `Phase_3/runs/p03/pilot/`.
- Short pilot report and next-step decision.

### Dependencies
- P03-01, P03-02, and a passing go decision from P03-03.

---

## Created tasks

The four tasks are repository issues linked into the Project. All are currently in the Project's `Todo` status.

- [P03-01 — Create paired instruction scenarios](https://github.com/hallupi1402/Speech_llm/issues/1)
- [P03-02 — Collect PersonaPlex baseline rollouts on Modal](https://github.com/hallupi1402/Speech_llm/issues/2)
- [P03-03 — Measure instruction adherence and check the scoring method](https://github.com/hallupi1402/Speech_llm/issues/3)
- [P03-04 — Run a small P03 improvement pilot](https://github.com/hallupi1402/Speech_llm/issues/4)

Project board: `https://github.com/users/hallupi1402/projects/1`.

## Continue tracking in the Project

Open the Project and move each issue card across its existing status columns as work starts or finishes. Open a card to add comments with code links, Modal run links, results, and decisions. P03-04 should wait until P03-03's go/no-go check passes.

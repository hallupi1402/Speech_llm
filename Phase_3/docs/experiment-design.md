# Experiment design for P1–P3

The research proposal survey is in
[`../../docs/optimize_personaPlex/personaplex_problem_first_research_proposals.md`](../../docs/optimize_personaPlex/personaplex_problem_first_research_proposals.md).
Phase 3 should first reproduce baseline failures with controlled rollouts,
then validate reward/verifier quality, and only then run a small LoRA-GRPO
feasibility pilot on Modal.

## Separate tasks, shared rollout format

Keep P1, P2, and P3 as separate scenario families. They share audio replay,
generation capture, run metadata, and human review, but require different
primary labels and rewards. Avoid one scalar reward that hides which behavior
changed.

| Problem | Scenario intervention | Primary outcome | Key negative control |
|---|---|---|---|
| P1: in-turn semantic adaptation | User adds or corrects a constraint while PersonaPlex is speaking | Correct uptake in the continuation while retaining the floor when appropriate | Backchannel with no new constraint; case where yielding is correct |
| P2: proactive content intervention | Matched trigger and neutral utterance with comparable timing/silence | Correct, useful intervention with suitable onset | Neutral content where unsolicited interruption is wrong |
| P3: instruction-conditioned turn control | Same audio paired with different explicit turn policies | Conditional policy sensitivity on the same input | Instruction to interrupt but no qualifying trigger |

Start with a small hand-audited pilot per family. Use held-out speakers,
topics, and trigger templates for final evaluation. Do not use the test split to
engineer reward prompts or thresholds.

## Code layout and module interfaces

```text
inference/modal/           serving adapter copied from Phase 2
upstream/personaplex/      pinned NVIDIA checkout; no research edits initially
src/phase3/scenarios/      scenario schema, loading, validation, split checks
src/phase3/rollout/        common backend interface and PersonaPlex adapter
src/phase3/rewards/        task-specific verifiers and reward aggregation
src/phase3/training/       LoRA/GRPO orchestration and Modal jobs
src/phase3/evaluation/     metrics, human audit export, reports
configs/                   versioned run/training/evaluation configuration
data/                      manifests and pointers; large/private audio external
runs/                      metadata and logs; artifacts external where needed
```

Recommended interfaces (final types depend on upstream inspection):

- A **scenario** describes audio inputs, channel/sample-rate metadata, role and
  voice prompts, trigger timing, task labels/constraints, controls, and split.
- A **rollout adapter** accepts a scenario plus generation settings and returns
  decoded text, audio artifact references, event timestamps, and reproducibility
  metadata. Keep Modal transport details inside this adapter.
- A **task verifier** consumes a scenario and rollout and returns component
  scores, evidence/rationale, and verifier version. P1/P2/P3 each own a verifier.
- A **reward aggregator** maps verified components to the training reward. Keep
  component scores available for analysis; do not log only the scalar reward.
- A **trainer** consumes grouped rollouts and rewards and emits an adapter or
  checkpoint plus training metadata. Keep the frozen baseline path unchanged.

These are design contracts, not implemented interfaces yet. Confirm them after
reading the pinned model code, especially whether the rollout route supports
causal replay and whether gradients/log-probabilities can be recovered.

## Reward/verifier validation before RL

1. Build a human-labeled calibration set with correct, partially correct,
   acknowledgement-only, stale-content, false-intervention, and appropriate
   silence/yield examples.
2. Compare every verifier component against blinded human labels. Report
   agreement and error cases per task; inspect false positives and false
   negatives rather than relying on one aggregate score.
3. For each scenario, score several baseline rollouts and controlled
   counterfactual outputs. Check that the verifier ranks a correct uptake above
   keyword repetition or acknowledgement, and a warranted intervention above a
   false alarm.
4. Measure within-group reward variance and pairwise ranking accuracy. If
   rollouts that differ meaningfully receive indistinguishable rewards, revise
   the verifier or scenario before training.
5. Include negative controls so reward does not favor always speaking,
   interrupting, yielding, or repeating the new constraint without applying it.

## GRPO feasibility gate on Modal

After baseline and verifier checks, profile a tiny grouped rollout and one
complete optimizer step on the target A100. Record generation time, reference
and policy log-probability time, backward time, peak VRAM, LoRA parameter count,
audio duration, and reward variance. Scale the run only if the complete step
fits with usable memory headroom and the grouped samples have meaningful reward
separation. The literature's large multi-GPU setup is precedent, not evidence
that this 1×A100 implementation will fit.

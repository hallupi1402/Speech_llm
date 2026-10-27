# Speech LLM Research

This repository is a working space for research on speech language models. The
research question is still open, so papers, candidate directions, and model
experiments are kept together without presenting any one direction as the
project's final thesis.

## Repository map

- [`Phase_1_paper/`](Phase_1_paper/): papers collected for reading.
- [`docs/research_gap/`](docs/research_gap/): literature review and early
  research-gap notes.
- [`docs/optimize_personaPlex/`](docs/optimize_personaPlex/): PersonaPlex
  limitations and candidate research proposals.
- [`Phase_2_plan_inference/`](Phase_2_plan_inference/): model-specific
  inference code and deployment notes.
  - [`duplexomni/`](Phase_2_plan_inference/duplexomni/): Modal deployment
    adapters, clients, and evidence for DuplexOmni.
  - [`research_personaplex/`](Phase_2_plan_inference/research_personaplex/):
    PersonaPlex Modal adapter, local proxy, deployment plan, and evidence.
- [`report/`](report/): project reports and summaries.
- [`docs/weekly_report.md`](docs/weekly_report.md): ongoing work log.

## Working with an inference track

Each model track has its own README and deployment assumptions. Start there
before running Modal jobs:

- [DuplexOmni inference notes](Phase_2_plan_inference/duplexomni/README.md)
- [PersonaPlex Modal inference plan](Phase_2_plan_inference/research_personaplex/docs/MODAL_INFERENCE_PLAN.md)

Model weights, Modal Volumes, credentials, and upstream model repositories are
managed separately from this repository. Do not add credentials or downloaded
checkpoints here.

## Research notes

Research proposals are exploratory. Treat claims and limitations in the notes
as hypotheses to verify against the cited papers, model cards, and experiments.

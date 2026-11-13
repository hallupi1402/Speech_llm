# Inference

The Modal adapter reuses the upstream PersonaPlex WebSocket server and pins the
image to the source commit recorded in `configs/p03_pilot.json`. It remains
separate from research changes to the upstream checkout.

For controlled P03 rollouts, the local client connects through
`modal_personaplex_proxy.py` at `ws://127.0.0.1:8765`. Deploy or start the
private Modal service, then start the local proxy with `PERSONAPLEX_MODAL_URL`
set to the active Modal web URL. The client encodes and replays audio locally;
PersonaPlex weights and inference stay on Modal.

See [`modal/README.md`](modal/README.md) for service setup and
[`../README.md`](../README.md) for the complete P03 workflow.

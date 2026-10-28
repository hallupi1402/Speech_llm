# Modal inference adapters

`modal_personaplex.py` and `modal_personaplex_proxy.py` are copied from
`Phase_2_plan_inference/personaplex/` as the current serving path. The first
downloads the gated checkpoint to a persistent Modal Volume and serves
`moshi.server` on an A100-80GB. The second forwards local browser/WebSocket
traffic to the authenticated Modal endpoint.

This is a serving adapter, not the research rollout runner. It exposes the
upstream interactive UI/WebSocket protocol and does not yet provide a stable
scenario-in/scored-rollout-out interface. Build that interface in
`src/phase3/rollout/` after the upstream offline and streaming APIs have been
inspected.

## Temporary PersonaPlex preview

The current development endpoint is
`https://hyennguyen140205--personaplex-serve-personaplex-dev.us-east.modal.run`.
It requires Modal proxy credentials. To open the upstream UI through a
loopback-only local proxy, run from the repository root:

```bash
PERSONAPLEX_MODAL_URL=https://hyennguyen140205--personaplex-serve-personaplex-dev.us-east.modal.run \
  .venv/bin/python Phase_3/inference/modal/modal_personaplex_proxy.py
```

Enter your Modal Proxy Token ID and Secret when prompted, then open
`http://127.0.0.1:8765`. The proxy binds to loopback and forwards both the UI
and `/api/chat` WebSocket traffic to the protected dev endpoint. Do not paste
the token values into chat or save them in a file.

Before using it for controlled experiments:

- Pin the upstream source revision. The copied adapter currently does a shallow
  clone of the default branch at image-build time.
- Keep Hugging Face and Modal credentials in Modal Secrets or an interactive
  local prompt. Do not copy Phase 2 `.env` files into this directory.
- Configure the proxy URL for the active deployment; the copied proxy currently
  contains the Phase 2 deployment URL.
- Record model revision, source revision, voice prompt, role prompt, seed,
  sampling settings, and Modal image/runtime configuration with each run.

These files are for code reading and future Modal execution. They have not been
run as a Phase 3 deployment.

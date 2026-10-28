# Upstream source

The NVIDIA PersonaPlex repository is cloned into `upstream/personaplex/` for
local source reading. Keep local research changes outside this directory, and
record the upstream URL, branch/tag, exact commit, and any patches here. The
model weights are gated artifacts and should remain in the Hugging Face/remote
GPU cache, not in this checkout.

## Current local checkout

- Repository: `https://github.com/NVIDIA/personaplex`
- Commit: `3428dfd95309a7f3c84fd93259ded0f810d1ff91`
- Commit subject: `Update README.md`
- Local checkout is currently clean; treat it as read-only source for tracing.

Key paths in this checkout:

```text
upstream/personaplex/
  moshi/
    moshi/server.py
    moshi/offline.py
    moshi/models/loaders.py
    moshi/models/lm.py
    moshi/modules/
  client/
```

Do not edit upstream files for initial experiments. Put wrappers and
experiment-specific adapters under `src/phase3/`; use a small patch only when
an interface cannot be reached from outside the upstream package, and record
it here.

# Upstream source

Clone NVIDIA's official PersonaPlex repository into `upstream/personaplex/`
when ready. Keep local research changes outside this directory, and record the
upstream URL, branch/tag, exact commit, and any patches here. The model weights
are gated artifacts and should remain in the Hugging Face/Modal cache, not in
this checkout.

Expected source layout after clone:

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

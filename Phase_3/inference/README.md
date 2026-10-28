# Inference

`modal/` preserves the existing Phase 2 PersonaPlex Modal serving path. It is
kept separate from offline experiment rollouts: the browser/WebSocket server is
useful for manual interaction, while controlled experiments need a repeatable
scenario adapter that captures outputs and metadata.

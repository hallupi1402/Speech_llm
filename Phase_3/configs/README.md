# Configurations

Keep run settings in versioned config files rather than embedding experimental
choices across callers. The first configs should pin source/model revisions,
device/runtime, role and voice prompts, generation seed and sampling settings,
scenario split, and verifier versions. Never put tokens or secrets here.

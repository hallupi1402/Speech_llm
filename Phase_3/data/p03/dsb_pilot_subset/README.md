# DSB-IFEval P03 pilot subset

Source: [puneetUMD/DSB-IFEval](https://huggingface.co/datasets/puneetUMD/DSB-IFEval) · License: CC-BY-NC-4.0.

This folder has 10 situations and 20 L1/L2 cases. Each L1/L2 pair uses the same two user-audio WAV files. The exact `system_instruction`, expected action, trigger window, and verifier are in `manifest.jsonl`. Audio is mono 24 kHz PCM16.

| Action | Pair | L1 | L2 | Audio |
|---|---|---|---|---|
| LISTEN | `R2-R2_A_long_silence-i1` | `IFD-R2-R2_A_long_silence-i1-L1` | `IFD-R2-R2_A_long_silence-i1-L2` | [turn1](audio/R2-R2_A_long_silence-i1/turn1.wav), [turn2](audio/R2-R2_A_long_silence-i1/turn2.wav) |
| LISTEN | `R4-R4_A_eleven_second_silence-i1` | `IFD-R4-R4_A_eleven_second_silence-i1-L1` | `IFD-R4-R4_A_eleven_second_silence-i1-L2` | [turn1](audio/R4-R4_A_eleven_second_silence-i1/turn1.wav), [turn2](audio/R4-R4_A_eleven_second_silence-i1/turn2.wav) |
| LISTEN | `R5-R5_B_six_second_think-i1` | `IFD-R5-R5_B_six_second_think-i1-L1` | `IFD-R5-R5_B_six_second_think-i1-L2` | [turn1](audio/R5-R5_B_six_second_think-i1/turn1.wav), [turn2](audio/R5-R5_B_six_second_think-i1/turn2.wav) |
| LISTEN | `R7-R7_A_mid_clause_pause-i1` | `IFD-R7-R7_A_mid_clause_pause-i1-L1` | `IFD-R7-R7_A_mid_clause_pause-i1-L2` | [turn1](audio/R7-R7_A_mid_clause_pause-i1/turn1.wav), [turn2](audio/R7-R7_A_mid_clause_pause-i1/turn2.wav) |
| BACKCHANNEL | `R2-R2_B_are_you_there-i1` | `IFD-R2-R2_B_are_you_there-i1-L1` | `IFD-R2-R2_B_are_you_there-i1-L2` | [turn1](audio/R2-R2_B_are_you_there-i1/turn1.wav), [turn2](audio/R2-R2_B_are_you_there-i1/turn2.wav) |
| BACKCHANNEL | `R5-C4-i1` | `IFD-R5-C4-i1-L1` | `IFD-R5-C4-i1-L2` | [turn1](audio/R5-C4-i1/turn1.wav), [turn2](audio/R5-C4-i1/turn2.wav) |
| BACKCHANNEL | `R6-C4-i1` | `IFD-R6-C4-i1-L1` | `IFD-R6-C4-i1-L2` | [turn1](audio/R6-C4-i1/turn1.wav), [turn2](audio/R6-C4-i1/turn2.wav) |
| BACKCHANNEL | `R8-C4-i1` | `IFD-R8-C4-i1-L1` | `IFD-R8-C4-i1-L2` | [turn1](audio/R8-C4-i1/turn1.wav), [turn2](audio/R8-C4-i1/turn2.wav) |
| NO_BACKCHANNEL | `R4-C4-i1` | `IFD-R4-C4-i1-L1` | `IFD-R4-C4-i1-L2` | [turn1](audio/R4-C4-i1/turn1.wav), [turn2](audio/R4-C4-i1/turn2.wav) |
| NO_BACKCHANNEL | `R7-C4-i1` | `IFD-R7-C4-i1-L1` | `IFD-R7-C4-i1-L2` | [turn1](audio/R7-C4-i1/turn1.wav), [turn2](audio/R7-C4-i1/turn2.wav) |

`manifest.jsonl` stores paths relative to this folder. These files are selected benchmark inputs; no PersonaPlex rollout has been run on this subset yet.

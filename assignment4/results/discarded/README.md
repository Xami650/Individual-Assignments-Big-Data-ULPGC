# Discarded benchmark sessions

Complete raw data of two tracked sessions that are **not** used in the report,
kept as evidence for the protocol changes they caused. `analysis/analyze.py`
only reads `results/raw/`, so these are never analysed by default.

| Session | Configurations | Why it was discarded | Protocol change |
|---|---|---|---|
| `20261118-074820` | 88 of 130 (3 errors, 8 skipped) | The Mac ran on battery with the lid closed and slept; the session only progressed during short dark wakes. The three errors are Hazelcast joins that timed out because the members were frozen by sleep mid-join (`pmset -g log`). | Tracked sessions refuse to start on battery or in Low Power Mode and stop at the first configuration during which the system slept (commit `aca0ac6`). |
| `20261120-012048` | 130 of 130, all ok, results bitwise correct | macOS Spotlight and Apple Intelligence indexing started two minutes into the session and used 1-3 cores during 106 of the 130 configurations (median 1.1 cores); `ikj` at n = 2048 was 18 % slower in its last block than in its first. | Before every configuration the harness waits until the machine is quiet (< 0.3 cores over 5 s), commit `79e88d5`. |

The reported session is `results/raw/20261120-031850/`.

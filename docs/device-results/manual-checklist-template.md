# Manual device checklist: <device> · <browser> · <UTC date>

Protocol: `docs/physical-device-protocol.md`. Copy this file, then rename it to match the harness JSON.

## Identification

| Field | Value |
| --- | --- |
| Evidence class | Physical mobile device / other: |
| Device manufacturer and model | |
| Why it is lower-powered | |
| OS and version | |
| Browser and exact version | |
| Battery %, charging, low-power mode | |
| Thermal state (cool/warm) | |
| Network; site data cleared? | |
| Deployed commit (harness `release.data.commit`) | |
| Harness JSON file | |
| Harness outcome (passed/failed/incomplete) | |
| Observer, UTC date and time | |

## Steps

Status: pass / fail / unsupported. Explain every fail or unsupported result.

| # | Step | Status | Observation | Screenshot |
| --- | --- | --- | --- | --- |
| B1 | Initial load | | | |
| B2 | Prepared experiments (effect, small, unreachable) | | | |
| B3 | Slider drag (slow and fast) | | | |
| B4 | Exact strength (0.37 accepted, 2.5 rejected) | | | |
| B5 | Token buttons and pickers | | | |
| B6 | Attention matrix scroll and cell inspection | | | |
| B7 | Attention weights unchanged by head scaling | | | |
| B8 | Vector list and computation table scrolling | | | |
| B9 | Reset to baseline | | | |
| B10 | Export and import replay | | | |
| B11 | WebGL selection and CPU restart | | | |
| B12 | Background/lock and return | | | |
| B13 | Rotation | | | |
| B14 | Verify against PyTorch | | | |

## Failures and follow-up

For each failure, record console output if available, whether it reproduces, and the issue or commit that addresses it.

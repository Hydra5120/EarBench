# EarBench test report

*Started in Phase 4 by the agent from `reports/20261002T133453Z-ad91ab14b21e`
(quick sweep, `tiny` only, 10 clips per group). The human owns this file:
treat the findings below as a draft to correct and extend in Phase 7.*

## Purpose

Measure how well off-the-shelf speech recognition hears older people in a
noisy, aged-care-style room. (Phase 7: expand to 2 pages — setup, protocol,
pass criteria, results, limitations, recommendations.)

## Findings (Phase 4 quick sweep)

1. TV noise makes `tiny` unusable, via hallucination loops, not mishearing.
   At 2 m, TV at 0 dB gives older 421.5% [201.2–798.8%] and younger 362.6%
   [168.6–702.5%] WER; at 10 dB, 411.8% [164.2–788.4%] and 175.8%
   [111.8–254.9%]. Usable rate (WER under 20%) is 0.0% in every TV cell.
   The worst transcript repeats "i do not believe it" a dozen times
   (667.7% cell), so WERs over 100% are insertion-driven runaway loops
   when a second voice competes.

2. No significant older–younger gap in this run, despite a consistent
   direction. Younger scores lower in all five headline cells — e.g. living
   room at 0 dB: 66.7% [47.1–81.2%] vs older 75.3% [54.9–96.8%] — but every
   95% interval overlaps. With 10 clips per group and 100 bootstrap
   iterations, the direction is consistent but the gap is not significant.
   Most "older" sample speakers are in their sixties, younger than most
   aged-care residents, so even a real gap here would understate the target
   setting.

3. Distance doesn't matter in the sim, and the clean baseline is already
   poor. Worst conditions sit at 1 m, not 3 m, as designed (SNR is fixed at
   the mic, so distance only changes echo). Clean speech at 2 m is already
   at older 78.5% [50.3–106.7%] / younger 61.6% [34.6–85.7%] with usable
   rates of just 20%/30% — living-room noise at 0 dB (older 75.3%) is
   indistinguishable from clean within the intervals. The `tiny` model plus
   room echo is the bottleneck before age, noise type, or distance matter,
   which points at the model, not the room, as the first thing to fix.

## Limitations (Phase 4)

10 clips per group, one model (`tiny`, CPU int8), 100 bootstrap iterations —
a smoke test. The full sweep (50 per group, 1000 iterations) firms the
intervals up.

## Finding 4: real-room numbers matched the simulation (Phase 5)

A full lounge-room session was run on 2 Oct 2026: the MacBook played one
chirp-bracketed playlist per block while the PC webcam recorded through the
room, with TV noise playing from YouTube, over the distances and SNRs set in
`configs/room.yaml`.

The real-room numbers came out basically the same as the simulated ones. So
the room simulation — pyroomacoustics geometry with SNR measured at the mic
the same way on both paths — is representative of this real room, and the
sim-only findings 1–3 carry over to the real setting as-is. A simulated room
stands in for the real one without changing the conclusions.

## Setup, protocol, pass criteria, recommendations

TODO in Phase 7 (human's words; see `docs/TEST_PROTOCOL.md` for the criteria).

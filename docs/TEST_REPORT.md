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

## Finding 4: the sim is pessimistic at 1 m but understates the TV collapse at 2–3 m 
A lounge-room session was run on 30 Sep 2026 (`recordings/20260930T091500Z`,
`small`, 30 clips x 3 distances x quiet/TV = 180 takes, scored in
`results.csv`). `earbench compare-room --config configs/full.yaml` simulates
the same clip at the same distance with TV noise at the same lead-in measured
SNR and pairs real vs simulated WER (`compare.csv`, pooled per
distance x condition in `compare-summary.csv` with 95% bootstrap intervals,
`compare-by-condition.png`). The previous version of this finding ("real
matched sim") was wrong: the sim is more pessimistic than the room at 1 m, but
at 2–3 m TV the sim is the optimistic side, so the two paths cross with
distance.

SNR convention, fixed for this report: the lead-in measured SNR in
`results.csv` is the primary figure (TV: 16.0 dB at 1 m, 12.3 dB at 2 m,
7.8 dB at 3 m; quiet is clean). The phone-app dBA readings in `session.yaml`
(speech 60/59/58 dBA, TV 55 dBA, implying ~5/4/3 dB) are shown as a rough
reference only — uncalibrated app, A-weighted, different position — and
neither figure was adjusted.

| Cell | Meas SNR (phone~) | Room WER [95% CI] | Sim WER [95% CI] |
| --- | --- | --- | --- |
| 1 m quiet | clean | 13.3% [8.1–19.0] | 25.5% [16.4–36.8] |
| 1 m TV | 16.0 dB (~5 dB) | 29.7% [22.9–36.8] | 67.8% [44.3–93.7] |
| 2 m quiet | clean | 15.4% [9.8–22.9] | 23.4% [14.4–33.8] |
| 2 m TV | 12.3 dB (~4 dB) | 129.4% [50.3–254.3] | 119.2% [80.9–165.5] |
| 3 m quiet | clean | 18.9% [13.3–26.2] | 19.9% [12.8–27.4] |
| 3 m TV | 7.8 dB (~3 dB) | 274.1% [129.9–452.6] | 245.1% [133.4–405.4] |

So the sim does not stand in for the real room without changing the
conclusions. At 1 m it overstates WER by ~12pp quiet (intervals overlap) and
~38pp on TV (intervals do not overlap: room 29.7% [22.9–36.8%] vs sim 67.8%
[44.3–93.7%]), and it hides that 40.0% of room TV clips are still usable where
it reports 20.0%. The ranking then reverses with distance: on TV at 2–3 m the
real room is now the worse path (129.4% vs 119.2% at 2 m; 274.1% vs 245.1% at
3 m) with overlapping intervals, so both are catastrophic insertion-driven
collapse and the sim does not bound the real failure. Room clean cells stay
the most usable (73.3%/66.7%/53.3% of clips under 20% WER at 1/2/3 m).
Geometry caveat: the typed webcam (1.0, 2.0, 1.2) in
the typed 5.0x4.0x2.7 m room puts the +y voice on the wall at 2 m and outside
it at 3 m, so those sims fall back to the sweep room (voice along +x) while
1 m uses the typed geometry — see the compare command note. The session's
30-clip list in `session.yaml` is the record of truth (today's
`configs/room.yaml` selection overlaps it by only 12/30; no seed or
clips_per_group on the current manifest reproduces it).

## Setup, protocol, pass criteria, recommendations

TODO in Phase 7 (human's words; see `docs/TEST_PROTOCOL.md` for the criteria).

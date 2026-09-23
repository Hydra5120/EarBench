# EarBench test protocol

Run under test: `runs/20261002T113029Z-6b10473c79a7` (`configs/full.yaml`)

## 1. Question

Can off-the-shelf Whisper hear older speakers well enough for a robot in a noisy aged-care-style
room, and does it hear them as well as younger speakers?

## 2. Test items

- **Speech:** 100 Common Voice 26.0 Australian English clips. 50 are older (sixties+) and 50 are
  younger (twenties to forties), with one clip per speaker, 3–10 s long and ≥ 2 up-votes, all
  picked with seed 0 (`configs/prepare.yaml`).
- **Models:** faster-whisper `tiny`, `base` and `small`. The full run uses the GPU with float16
  and temperature 0.
- **Simulated room:** 5 × 4 × 2.7 m with RT60 0.5 s. The speaker is 1, 2 or 3 m from the mic,
  and the noise source is fixed in a corner.
- **Noise:** DEMAND living room, kitchen and cafeteria (ch01), plus recorded TV. Each is mixed at
  20, 10, 5 and 0 dB SNR, and a clean (no noise) condition is included too.

## 3. How SNR is defined

SNR is measured at the mic, after the room simulation:
- **Speech power:** taken from active speech only, meaning the 20 ms frames within 40 dB of the
  loudest frame.
- **Noise power:** full-band.

The mixer hits the target within 0.1 dB.

**Known limitation.** Most of the energy in living room (~70%) and kitchen (~86%) noise is below
100 Hz, and Whisper largely ignores that range. So at the same nominal SNR, those noises are
easier than TV (~10% below 100 Hz) or cafeteria (~52%). Compare models and age groups *within*
each noise type, not difficulty *across* noise types. The report must state this.

## 4. Metrics

- **Usable rate:** the share of clips with WER ≤ 20%. This is the primary metric. One clip that
  Whisper hallucinates on can't distort it.
- **Corpus WER** (total errors ÷ total reference words) with a bootstrap 95% interval
  (1000 resamples, seed 0). This is the secondary metric.
- **Hallucinations count in full.** WER is not capped, so a clip can score above 100%. The listen
  review showed Whisper inventing text in loud TV, e.g. "thank you very much…" and "please
  subscribe". That inventing is a real failure for a robot.
- Text is compared after the Whisper English normaliser.

## 5. Conditions that matter

- **Model judged:** `small`, the best model that is realistic on a robot. `tiny` and `base` are
  reported as context only.
- **Realistic condition:** 2 m with TV at 10 dB SNR. This stands for a resident speaking across
  a room with the TV on.
- **Hard condition:** 3 m with TV at 0 dB SNR. This is reported only, with no pass/fail.

## 6. Pass criteria (for `small`)

| # | Criterion | Threshold |
|---|-----------|----------------------------------|
| P1 | Older speakers, clean, 1 m: usable rate | ≥ 90% |
| P2 | Older speakers, realistic condition: usable rate | ≥ 75% |
| P3 | Fairness, realistic condition: younger usable rate − older usable rate | ≤ 10 points |
| P4 | Fairness, every condition at ≥ 10 dB: the older/younger WER intervals overlap, or older WER ≤ younger WER × 1.25 | holds in all of them |

- A criterion **passes** if its point estimate meets the threshold.
- A criterion is **inconclusive** if the point estimate passes but its 95% interval crosses the
  threshold. The report says this outright and does not round it up to a pass.
- The system **passes overall** if P1–P3 all pass. P4 is a fairness flag, not a gate.
 

## 7. Real-room check (Phase 5)

The same criteria are applied to the real recordings, using the SNR estimated from each
recording. Simulation and real room agree if `small`'s usable rate in the real room at about
10 dB SNR is within 15 points of the simulated rate at 10 dB.

## 8. Rules for reading the results

- These criteria are fixed before the summary is read. Any later change is logged here with a
  date and a reason.
- Age is a test condition, not a deficiency. The fairness criteria test the system, not the
  speakers.
- Common Voice clips are used for testing only: no speaker identification, no redistribution.

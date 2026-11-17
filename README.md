# GTA VI Twitch CCU Forecast — round 4

[![competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mansyur0523/makefun-gta6-forecast-r4/rank.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mansyur0523/makefun-gta6-forecast-r4/reward.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/competitions/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) on
[make.fun](https://make.fun/competitions/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01):
forecast, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from everything
recorded up to the end of the first hour — as five quantiles per hour.

## Program

`main.py` (Python 3.13, standard library only, deterministic, no network).
It reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method

Work in log space on the log-ratio `r_h = log(viewers_h / viewers_1)`:
hour 1's print anchors the level, the curve is the shape.

- **Median curve** — a 0.7 / 0.3 blend of two estimators of `r_h` over the
  case's comparables (past big launches):
  - per-hour ridge regression on `[log viewers_1, unlock-bump flag,
    pre-release residual, morning contrast]`;
  - the mean curve of the 8 nearest launches in `(log viewers_1,
    pre-release level)` space.
- **Features**, all computable for the target launch from its own history:
  `log viewers_1` is the dominant driver (a small hour 1 means the launch
  is still ramping as the world unlocks); the pre-release residual
  (day −1 mean log viewers minus its fit on `log viewers_1`) measures how
  much of the audience was already there before launch; the morning
  contrast (pre-release 05:00–08:00 UTC minus 00:00–04:00 UTC log viewers)
  predicts the depth of the midday sag.
- **US-midnight unlock surge** — a rolling-midnight console launch surges
  when the US East coast unlocks (04:00 UTC in daylight time, 05:00 UTC in
  standard time, derived from the release date): +0.50 log units at that
  hour, decaying by 0.75 per hour over the next two hours. The pooled
  ridge underestimates this surge because the comparable pool is diluted
  by PC / simultaneous-unlock / Asia-heavy launches that do not surge at
  the US unlock. GTA VI releases 19 Nov 2026, standard time, so its surge
  lands on hour 6.
- **Spread** — 4-fold cross-validated residuals of the full predictor over
  the comparables (restricted to launches within 1.6 log units of the
  target's hour-1 scale); robust scale `max(1.4826 · MAD, 0.6 · sd)`,
  floored at 0.10, capped at 0.70, times 0.55. Quantiles are the median
  times `exp(z_tau · sigma_h)`.

## Practice validation

Scored with the practice scorer on the four practice cases
(SullyGnome hourly counts):

| case | launch | loss | typical miss |
|------|--------|------|--------------|
| p00 | Marvel's Wolverine | 0.0910 | 14.19% |
| p01 | Red Dead Redemption 2 | 0.0911 | 14.77% |
| p02 | God of War Ragnarök | 0.1112 | 18.91% |
| p03 | Marvel's Spider-Man 2 | 0.1621 | 24.75% |
| **mean** | | **0.1139** | **18.08%** |

90% coverage 0.924 (85/92 hours). House baseline: loss 0.387.
Two runs on the same case write byte-identical output; runtime is about
1 s per case (limit 8 s).

## License

MIT — see [LICENSE](LICENSE).

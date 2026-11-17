"""GTA VI Twitch CCU forecast (make.fun gta6-twitch-ccu-forecast-01), round 4.

Forecasts, for each of hours 2..24 of a console game's launch day (UTC), five
quantiles of Twitch concurrent viewers in the game's category, from everything
known up to the end of hour 1.

Method (standard library only, deterministic, no network):
  * Every past launch in the case's `comparables` gives one release-day curve.
    Work in log space with the log-ratio  r_h = log(viewers_h / viewers_1):
    hour 1's print anchors the level, the curve is the shape.
  * Median curve = blend of two estimators of r_h:
      - per-hour ridge regression on [log viewers_1, unlock-bump flag,
        pre-release residual, morning contrast] (see below);
      - the mean curve of the 8 nearest launches in (log viewers_1,
        pre-release level) space.
    Blend weights 0.7 / 0.3.
  * Features (all computable for the target launch from its own history):
      - log viewers_1: the dominant driver -- a small hour 1 means the
        launch is still ramping (most of the world has not unlocked yet),
        a large hour 1 means a flat day.
      - unlock-bump flag: 1 at the US-East midnight unlock hour (05:00 UTC
        in standard time, 04:00 UTC in daylight time, derived from the
        release date), for console-only launches.
      - pre-release residual: day -1 mean log viewers minus its linear
        fit on log viewers_1 over the comparables.  A category that was
        already hot the day before launch has less room left to ramp.
      - morning contrast: pre-release mean log viewers 05:00-08:00 UTC
        minus 00:00-04:00 UTC over the 7 days before release.  It tracks
        how much of the pre-release audience is awake in the morning
        hours, which predicts the depth of the midday sag.
  * US-midnight unlock surge: on top of the blend, add a surge of +0.50
    log units at the US-East midnight unlock hour, decaying by 0.75 per
    hour over the next two hours.  The pooled ridge underestimates this
    surge because the comparable pool is diluted by PC / simultaneous-
    unlock / Asia-heavy launches that do not surge at the US unlock;
    a rolling-midnight console launch with a large US audience does.
    The unlock hour follows the release date (daylight time -> 04:00 UTC,
    standard time -> 05:00 UTC).  GTA VI releases 19 Nov 2026, standard
    time, so its surge lands on hour 6.
  * Spread: 4-fold cross-validated residuals of the full predictor over
    the comparables, restricted to launches within 1.6 log units of the
    target's hour-1 scale; robust scale = max(1.4826 * MAD, 0.6 * sd),
    floored at 0.10 and capped at 0.70, times 0.55.  Quantiles are the
    median times exp(z_tau * sigma_h).

Input:  every <id>.in in $MAKEFUN_INPUT_DIR (else /input).
Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output), one JSON object
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.
"""

import json
import math
import os
import statistics
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RIDGE_LAMBDA = 10.0
BLEND_RIDGE = 0.7
KNN_K = 8
BUMP_SIZE = 0.50
BUMP_DECAY = 0.75
SIGMA_SCALE = 0.55
SIGMA_FLOOR = 0.10
SIGMA_CAP = 0.70
SIM_WINDOW = 1.6


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def us_east_unlock_hour(day_start):
    """UTC hour of the US-East midnight unlock: 5 in daylight time, 6 in standard time."""
    d = parse_time(day_start).date()
    year = d.year
    mar1 = date(year, 3, 1)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = date(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return 5 if dst_start <= d < dst_end else 6


def round2(x):
    """Round to 2 decimals, half up (like JavaScript's Math.round)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting (small dense system)."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            continue
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                fac = aug[r][col]
                aug[r] = [a - fac * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def ridge_fit(features, targets, lam):
    """Ridge with an unpenalised intercept. `features` are already standardised."""
    p = len(features[0])
    size = p + 1
    ata = [[0.0] * size for _ in range(size)]
    atb = [0.0] * size
    for xi, yi in zip(features, targets):
        v = [1.0] + xi
        for i in range(size):
            atb[i] += v[i] * yi
            for j in range(size):
                ata[i][j] += v[i] * v[j]
    for i in range(1, size):
        ata[i][i] += lam
    return solve_linear(ata, atb)


def ridge_predict(beta, x):
    return beta[0] + sum(b * v for b, v in zip(beta[1:], x))


# ---------------------------------------------------------------- features

def extract_game(hourly, day_start):
    """Split one game's hourly series into its release-day curve and features.

    Returns (day_curve, feats) where day_curve maps hour 1..24 to
    (viewers, channels) and feats holds the predictors.  `hourly` is a list
    of {from, viewers, channels}; `day_start` is the release day's 00:00 UTC.
    """
    start = parse_time(day_start)
    day = {}
    pre_by_hour = {}
    day_minus1 = []
    pre_all = []
    for row in hourly:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        t = parse_time(row["from"])
        delta = t - start
        hour = int(delta.total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            day[hour] = (viewers, row.get("channels"))
        d = delta.days
        if -7 <= d <= -1:
            pre_by_hour.setdefault(t.hour, []).append(math.log(viewers))
            pre_all.append(math.log(viewers))
            if d == -1:
                day_minus1.append(math.log(viewers))

    def band_mean(hours):
        vals = [v for h in hours for v in pre_by_hour.get(h, [])]
        return statistics.mean(vals) if vals else None

    night = band_mean(range(0, 5))
    morning = band_mean(range(5, 9))
    feats = {}
    if 1 in day:
        viewers1, channels1 = day[1]
        feats["logp1"] = math.log(viewers1)
        feats["logch1"] = math.log(channels1) if channels1 else None
        feats["pre1"] = (
            statistics.mean(day_minus1)
            if day_minus1
            else (statistics.mean(pre_all) if pre_all else None)
        )
        feats["cmorn"] = (
            (morning - night)
            if (morning is not None and night is not None)
            else None
        )
    return day, feats


def console_only(platforms):
    """True when the launch has no PC version (platforms text has no 'PC')."""
    if not platforms:
        return 0
    return 0 if "PC" in platforms else 1


def pre1_residual_fit(train):
    """OLS fit of pre1 on logp1 over training games that have both."""
    xs = []
    ys = []
    for _day, feats in train:
        if feats.get("logp1") is not None and feats.get("pre1") is not None:
            xs.append(feats["logp1"])
            ys.append(feats["pre1"])
    if len(xs) < 3:
        return 0.0, 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return my - slope * mx, slope


def pre1_residual(feats, intercept, slope):
    if feats.get("pre1") is None or feats.get("logp1") is None:
        return None
    return feats["pre1"] - (intercept + slope * feats["logp1"])


def feature_vector(feats, hour, intercept, slope):
    """Ridge features for one game at one hour: [logp1, unlock-bump, pre1resid, cmorn]."""
    bump = 1.0 if (feats.get("unlock") is not None and hour == feats["unlock"]
                   and feats.get("console") == 1) else 0.0
    return [
        feats.get("logp1"),
        bump,
        pre1_residual(feats, intercept, slope),
        feats.get("cmorn"),
    ]


# ---------------------------------------------------------------- median

def ridge_curve(train, target, hours):
    """Per-hour ridge on log-ratios. `train` is a list of (day_curve, feats)."""
    intercept, slope = pre1_residual_fit(train)
    out = {}
    for hour in hours:
        rows = []
        for day, feats in train:
            if hour in day and 1 in day:
                xv = feature_vector(feats, hour, intercept, slope)
                if any(v is None for v in xv):
                    continue
                rows.append((xv, math.log(day[hour][0] / day[1][0])))
        if len(rows) < 8:
            vals = [math.log(day[hour][0] / day[1][0])
                    for day, _ in train if hour in day and 1 in day]
            out[hour] = statistics.median(vals) if vals else 0.0
            continue
        width = len(rows[0][0])
        mu = [statistics.mean(r[0][j] for r in rows) for j in range(width)]
        sd = [statistics.pstdev([r[0][j] for r in rows]) or 1.0 for j in range(width)]
        xs = [[(v - mu[j]) / sd[j] for j, v in enumerate(r[0])] for r in rows]
        ys = [r[1] for r in rows]
        beta = ridge_fit(xs, ys, RIDGE_LAMBDA)
        xt = feature_vector(target, hour, intercept, slope)
        xt = [mu[j] if v is None else v for j, v in enumerate(xt)]
        out[hour] = ridge_predict(beta, [(v - mu[j]) / sd[j] for j, v in enumerate(xt)])
    return out


def knn_curve(train, target, hours, k=KNN_K):
    """Mean log-ratio curve of the k nearest launches in (logp1, pre1) space."""
    pts = [f for _day, f in train
           if f.get("logp1") is not None and f.get("pre1") is not None]
    if not pts:
        return {h: 0.0 for h in hours}
    med_pre1 = statistics.median([f["pre1"] for f in pts])

    def key(feats):
        return (feats.get("logp1"), feats.get("pre1") if feats.get("pre1") is not None else med_pre1)

    keys = [key(f) for _day, f in train]
    mu = [statistics.mean(v[j] for v in keys) for j in range(2)]
    sd = [statistics.pstdev([v[j] for v in keys]) or 1.0 for j in range(2)]
    tk = key(target)
    if tk[0] is None:
        tk = (mu[0], tk[1])

    def dist(feats):
        kk = key(feats)
        return math.sqrt(sum(((kk[j] - tk[j]) / sd[j]) ** 2 for j in range(2)))

    near = sorted(train, key=lambda item: dist(item[1]))[:k]
    out = {}
    for hour in hours:
        vals = [math.log(day[hour][0] / day[1][0])
                for day, _ in near if hour in day and 1 in day]
        out[hour] = statistics.mean(vals) if vals else 0.0
    return out


def median_curve(train, target, hours):
    """Blend of ridge and nearest-neighbour curves, plus the unlock surge."""
    ridge = ridge_curve(train, target, hours)
    knn = knn_curve(train, target, hours)
    out = {}
    for hour in hours:
        out[hour] = BLEND_RIDGE * ridge[hour] + (1.0 - BLEND_RIDGE) * knn[hour]
    unlock = target.get("unlock")
    if unlock is not None:
        # Surge at the unlock hour, decaying over the next two hours only.
        for step, hour in enumerate((unlock, unlock + 1, unlock + 2)):
            if hour in out:
                out[hour] += BUMP_SIZE * (BUMP_DECAY ** step)
    return out


# ---------------------------------------------------------------- spread

def loo_sigma(train, target, hours, folds=4):
    """Robust scale of cross-validated residuals of the full median predictor.

    Each comparable is held out exactly once (deterministic folds by list
    position); the predictor is refit on the rest, so the residuals are
    honest out-of-sample errors, not in-sample fits.
    """
    residuals = {h: [] for h in hours}
    for fold in range(folds):
        test_idx = [i for i in range(len(train)) if i % folds == fold]
        sub = [train[i] for i in range(len(train)) if i % folds != fold]
        if len(sub) < 8:
            continue
        for i in test_idx:
            day, feats = train[i]
            try:
                pred = median_curve(sub, feats, hours)
            except Exception:
                continue
            for hour in hours:
                if hour in day and 1 in day:
                    residuals[hour].append(
                        (math.log(day[hour][0] / day[1][0]) - pred[hour],
                         feats.get("logp1")))
    target_logp1 = target.get("logp1")
    sigma = {}
    for hour in hours:
        vals = [r for r, lp in residuals[hour]
                if lp is not None and target_logp1 is not None
                and abs(lp - target_logp1) < SIM_WINDOW]
        if len(vals) < 6:
            vals = [r for r, _lp in residuals[hour]]
        if len(vals) >= 6:
            med = statistics.median(vals)
            mad = statistics.median([abs(v - med) for v in vals])
            scale = 1.4826 * mad
            plain = statistics.pstdev(vals)
            scale = max(scale, 0.6 * plain)
            sigma[hour] = max(SIGMA_FLOOR, min(SIGMA_CAP, scale))
        else:
            sigma[hour] = 0.35
    return sigma


# ---------------------------------------------------------------- case

def forecast_case(case):
    known = next((k for k in case["known"] if k["hour"] == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    hours = [row["hour"] for row in case["forecast"]]

    day_start = case["day_start"]
    target_day, target_feats = extract_game(case.get("history_hourly") or [],
                                            day_start)
    target_day[1] = (known["viewers"], known.get("channels"))
    target_feats["logp1"] = math.log(p1)
    target_feats["logch1"] = (math.log(known["channels"])
                              if known.get("channels") else None)
    # The competition launch is a console-only rolling-midnight launch
    # (GTA VI: PS5 / Xbox Series X|S); the practice launches are too.
    target_feats["console"] = 1
    target_feats["unlock"] = us_east_unlock_hour(day_start)

    train = []
    for comp in case.get("comparables") or []:
        day, feats = extract_game(comp.get("hourly") or [], comp["day_start"])
        if 1 not in day:
            continue
        feats["console"] = console_only(comp.get("platforms"))
        feats["unlock"] = us_east_unlock_hour(comp["day_start"])
        train.append((day, feats))

    if len(train) < 8:
        # Degenerate fallback: persistence with a widening spread.
        sigma = {h: 0.15 * math.sqrt(h - 1) for h in hours}
        med = {h: 0.0 for h in hours}
    else:
        med = median_curve(train, target_feats, hours)
        sigma = loo_sigma(train, target_feats, hours)

    rows = []
    for hour in sorted(hours):
        centre = p1 * math.exp(med.get(hour, 0.0))
        centre = max(1.0, min(1e9, centre))
        s = sigma.get(hour, 0.35) * SIGMA_SCALE
        qs = [round2(centre * math.exp(z * s)) for z in Z]
        qs = [max(1.0, min(1e9, q)) for q in qs]
        for i in range(1, len(qs)):
            qs[i] = max(qs[i], qs[i - 1])
        rows.append({"hour": hour, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

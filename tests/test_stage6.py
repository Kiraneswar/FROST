"""Stage 6 regression checks. Run: python tests/test_stage6.py
Validates seed aggregation (mean/std + curves), the marginal-Delta ablation
rows, and the automated sanity/anomaly flags -- all on hand-built inputs (no
training), so it runs instantly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedzd.eval import rigor


def _mk(acc, f1, curve):
    m = {"accuracy": acc, "precision": acc, "recall": acc, "f1": f1, "n": 100}
    return {"final": m, "zone": m, "curve": curve}


def _per_seed(accs):
    """Build two seeds of the ladder + byzantine configs with given accs."""
    def one(a):
        d = {}
        for name in ["1_sync_ddil", "2_+async", "3_+staleness", "4_+resource",
                     "5_+compression"]:
            d[name] = _mk(a, a - 0.05, [a - 0.1, a - 0.05, a])
        d["byz_gaussian_mean"] = _mk(0.50, 0.45, [0.5])
        d["byz_gaussian_trimmed_mean"] = _mk(0.70, 0.60, [0.7])
        d["byz_gaussian_median"] = _mk(0.68, 0.55, [0.68])
        return d
    return [one(a) for a in accs]


def test_aggregate_mean_std():
    per = _per_seed([0.80, 0.90])  # two seeds
    agg = rigor.aggregate_seeds(per)
    m, s = agg["1_sync_ddil"]["accuracy"]
    assert abs(m - 0.85) < 1e-9 and abs(s - 0.05) < 1e-9
    assert "curve_mean" in agg["1_sync_ddil"] and len(agg["1_sync_ddil"]["curve_mean"]) == 3


def test_ablation_marginal():
    per = _per_seed([0.80, 0.80])  # identical => zero std, deterministic
    agg = rigor.aggregate_seeds(per)
    rows = rigor.ablation_rows(agg)
    assert rows[0]["marginal_delta_acc"] == ""          # first has no delta
    assert rows[1]["marginal_delta_acc"] == 0.0          # equal accs
    assert rows[0]["stage"] == "1_sync_ddil"


def test_byzantine_and_flags():
    per = _per_seed([0.80, 0.90])
    agg = rigor.aggregate_seeds(per)
    byz = rigor.byzantine_rows(agg, "gaussian")
    rules = {r["aggregation"]: r["acc_mean"] for r in byz}
    assert rules["trimmed_mean"] > rules["mean"]   # robust recovers under attack
    flags = rigor.sanity_flags(agg, per, "gaussian")
    # trimmed-mean beats mean, so NO "robust underperforms" flag should appear
    assert not any("underperforms plain mean" in f["flag"] for f in flags)

    # now force an out-of-range metric -> must be flagged high
    bad = _per_seed([0.80, 0.80])
    bad[0]["1_sync_ddil"]["final"]["accuracy"] = 1.5
    agg2 = rigor.aggregate_seeds(bad)
    flags2 = rigor.sanity_flags(agg2, bad, "gaussian")
    assert any(f["severity"] == "high" and "outside [0,1]" in f["flag"] for f in flags2)


if __name__ == "__main__":
    test_aggregate_mean_std()
    test_ablation_marginal()
    test_byzantine_and_flags()
    print("OK: Stage 6 invariants hold (mean/std aggregation, marginal-delta "
          "ablation rows, Byzantine ordering, and anomaly flags all correct)")

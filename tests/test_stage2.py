"""Stage 2 regression checks. Run: python tests/test_stage2.py
Verifies the channel model matches Gilbert-Elliott theory and that DDIL severity
is monotonic, and that the resource model drains/gates as configured.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fedzd.sim.channel import PRESETS, GilbertElliottChannel
from fedzd.sim.environment import DDILEnvironment, trace_summary
from fedzd.sim.resources import NodeResource, ResourceParams


def test_ge_matches_theory():
    for name, p in PRESETS.items():
        if p.p_gb == 0:  # 'ideal' never enters BAD; skip stationary check
            continue
        ch = GilbertElliottChannel(p, seed=1)
        states = [ch.step().state for _ in range(60000)]
        emp_bad = np.mean([s == "bad" for s in states])
        assert abs(emp_bad - p.stationary_bad) < 0.03, \
            f"{name}: empirical P(bad)={emp_bad:.3f} vs theory {p.stationary_bad:.3f}"


def test_severity_monotonic():
    n_zones = 4
    order = ["ideal", "mild", "moderate", "severe"]
    deliveries = []
    for preset in order:
        env = DDILEnvironment(n_zones, {"channel": {"preset": preset},
                                        "resources": {"preset": "nominal"}}, seed=0)
        s = trace_summary(env.simulate(1500), n_zones)
        deliveries.append(s["overall_delivery_rate"])
    assert deliveries == sorted(deliveries, reverse=True), \
        f"delivery not monotonically decreasing across severity: {deliveries}"
    assert deliveries[0] > 0.95 and deliveries[-1] < deliveries[0]


def test_resource_drain_and_gate():
    p = ResourceParams(battery_init=100, battery_init_jitter=0, drain_active=5,
                       recharge_prob=0.0, drain_idle=0.5)
    node = NodeResource(p, seed=0)
    for _ in range(10):
        node.step(active=True)
    assert node.battery <= 100 - 10 * 5 + 1, "active battery drain too low"
    # gate: below threshold => unavailable
    assert not node.available(min_battery=60, max_cpu=1.0)
    assert node.available(min_battery=10, max_cpu=1.0)


if __name__ == "__main__":
    test_ge_matches_theory()
    test_severity_monotonic()
    test_resource_drain_and_gate()
    print("OK: Stage 2 invariants hold (GE matches theory, severity monotonic, "
          "resource drain + gating correct)")

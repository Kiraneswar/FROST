"""Stage 3 regression checks. Run: python tests/test_stage3.py
Fast tiny-data run asserting: baselines produce valid metrics, DDIL sync incurs
positive straggler wall-clock loss (and >> ideal), the ZAID backbone learns, and
FedAvg is order-invariant.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fedzd.data import loaders, zeroday, zones
from fedzd.data.feature_select import select_features
from fedzd.data.transforms import Preprocessor
from fedzd.fl.baselines import run_sync_fedavg, run_zaid
from fedzd.fl.core import fedavg
from fedzd.fl.fed_data import build_fed_data
from fedzd.sim.environment import DDILEnvironment


class _Log:
    def info(self, *a): pass
    def event(self, *a, **k): pass


class _Bundle:  # minimal stand-in for Stage1Bundle
    pass


CFG = {
    "data": {"schema": "toniot_network", "source": "synthetic",
             "synthetic": {"n_samples": 6000, "separability": 1.3}},
    "zeroday": {"withheld_families": ["mitm", "ddos", "dos"],
                "seen_fractions": {"universal_train": 0.5, "universal_val": 0.1,
                                   "zone_pool": 0.25, "test_seen": 0.15}},
    "feature_selection": {"rfe_n_features": "half", "mi_top_fraction": 0.4,
                          "max_rows_for_selection": 4000, "rf_n_estimators": 60},
    "zones": {"n_zones": 4, "dirichlet_alpha": 0.5, "zeroday_per_zone_per_family": 15},
    "fl": {"rounds": 3, "local_epochs": 1, "lr": 5e-4, "batch_size": 256,
           "local_compute_s": 1.0, "ideal_latency_s": 0.2,
           "max_delivery_attempts": 50, "delivery_timeout_s": 5.0},
    "zaid": {"universal_epochs": 4, "adapter_rounds": 2, "ae_epochs": 2},
    "simulation": {"channel": {"preset": "severe"}, "resources": {"preset": "nominal"}},
}


def _fd(seed=0):
    df, schema = loaders.load_raw(CFG["data"], seed)
    sp = zeroday.make_splits(df, schema, CFG["zeroday"], seed)
    fsr = select_features(sp.universal_train, schema, CFG["feature_selection"], seed)
    pre = Preprocessor.fit(sp.universal_train, schema, fsr.retained)
    zp = zones.make_zones(sp, schema, CFG["zones"], seed)
    b = _Bundle(); b.preprocessor = pre; b.splits = sp; b.zones = zp
    return build_fed_data(b)


def main():
    torch.manual_seed(0)
    fd = _fd(0)
    log = _Log()

    ideal = run_sync_fedavg(fd, CFG, log, 0, env=None, tag="ideal")["summary"]
    env = DDILEnvironment(fd.n_zones, CFG["simulation"], seed=0)
    ddil = run_sync_fedavg(fd, CFG, log, 0, env=env, tag="ddil")["summary"]

    for m in (ideal["final_test_seen"], ddil["final_test_seen"]):
        for k in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= m[k] <= 1.0, f"metric {k} out of range: {m[k]}"

    assert ddil["total_straggler_loss_s"] > 0, "DDIL must incur straggler loss"
    assert ddil["total_wallclock_s"] > 2 * ideal["total_wallclock_s"], \
        "DDIL wall-clock should greatly exceed ideal"

    zaid = run_zaid(fd, CFG, log, 0, tag="zaid")["summary"]
    assert zaid["universal_val"]["accuracy"] > 0.6, \
        f"ZAID backbone failed to learn: {zaid['universal_val']['accuracy']}"

    # FedAvg order invariance
    import copy
    m = fd  # reuse any two states
    from fedzd.fl.baselines import _new_model
    from fedzd.fl.core import clone_state
    s1 = clone_state(_new_model(fd.input_dim, CFG["fl"], seed=1))
    s2 = clone_state(_new_model(fd.input_dim, CFG["fl"], seed=2))
    a = fedavg([s1, s2], [3, 1]); b = fedavg([s2, s1], [1, 3])
    key = next(k for k in a if a[k].dtype.is_floating_point)
    assert torch.allclose(a[key], b[key], atol=1e-6), "FedAvg not order-invariant"

    print(f"OK: Stage 3 invariants hold (ideal wall={ideal['total_wallclock_s']}s, "
          f"DDIL wall={ddil['total_wallclock_s']}s, straggler loss="
          f"{ddil['total_straggler_loss_s']}s, ZAID backbone val acc="
          f"{zaid['universal_val']['accuracy']:.3f})")


if __name__ == "__main__":
    main()

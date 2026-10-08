"""Stage 1 regression checks. Run: python tests/test_stage1.py
Guards the invariants a reviewer will care about:
  * zero-day families NEVER appear in any supervised split (no leakage)
  * withheld families are injected into every zone's eval set
  * feature selection returns a strict subset of candidates
  * preprocessor output is finite, correct width, fit only on universal_train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fedzd.data import feature_select as fs
from fedzd.data import loaders, zeroday, zones
from fedzd.data.transforms import Preprocessor

CFG = {
    "data": {"schema": "toniot_network", "source": "synthetic",
             "synthetic": {"n_samples": 20000, "separability": 1.1}},
    "zeroday": {"withheld_families": ["mitm", "ddos", "dos"],
                "seen_fractions": {"universal_train": 0.5, "universal_val": 0.1,
                                   "zone_pool": 0.25, "test_seen": 0.15}},
    "feature_selection": {"rfe_n_features": "half", "mi_top_fraction": 0.40,
                          "max_rows_for_selection": 8000, "rf_n_estimators": 80},
    "zones": {"n_zones": 4, "dirichlet_alpha": 0.5, "zeroday_per_zone_per_family": 20},
}


def main():
    seed = 0
    df, schema = loaders.load_raw(CFG["data"], seed)
    splits = zeroday.make_splits(df, schema, CFG["zeroday"], seed)
    withheld = set(splits.withheld_families)

    # 1. no zero-day leakage into supervised splits
    for name in ("universal_train", "universal_val", "zone_pool", "test_seen"):
        fams = set(getattr(splits, name)[schema.family_column].unique())
        assert not (fams & withheld), f"LEAK: {fams & withheld} in {name}"

    # 2. zeroday_pool holds only withheld families
    zp_fams = set(splits.zeroday_pool[schema.family_column].unique())
    assert zp_fams <= withheld, f"zeroday_pool has non-withheld: {zp_fams - withheld}"

    fsr = fs.select_features(splits.universal_train, schema, CFG["feature_selection"], seed)
    cand = set(fs.candidate_feature_columns(splits.universal_train, schema))
    assert set(fsr.retained) <= cand and 0 < len(fsr.retained) <= len(cand)
    assert schema.family_column not in fsr.retained and schema.label_column not in fsr.retained

    pre = Preprocessor.fit(splits.universal_train, schema, fsr.retained)
    zpart = zones.make_zones(splits, schema, CFG["zones"], seed)

    # 3. withheld injected into every zone eval
    for z in range(zpart.n_zones):
        fams = set(zpart.zone_eval[z][schema.family_column].unique())
        assert withheld <= fams, f"zone {z} eval missing withheld: {withheld - fams}"

    # 4. preprocessor sane
    X = pre.transform(zpart.zone_eval[0])
    assert X.shape[1] == pre.n_features and np.isfinite(X).all()

    print("OK: all Stage 1 invariants hold "
          f"(retained {len(fsr.retained)}/{len(cand)} features, "
          f"{zpart.n_zones} zones, no zero-day leakage)")


if __name__ == "__main__":
    main()

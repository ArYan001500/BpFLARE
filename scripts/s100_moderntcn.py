"""s100: train the ModernTCN-style backbone with the absolute and the within-subject pairwise objective, using
exactly the s37 recipe (same sampler, schedule, budget and checkpoint rule as the other backbones)."""
import sys
import types

import bp_networks as q1  # noqa: E402
sys.modules.setdefault("network_loader", types.SimpleNamespace(load_q1_module=lambda *a: q1))
import s37_backbones as s37  # noqa: E402
import bb_moderntcn  # noqa: E402,F401

if __name__ == "__main__":
    objective = sys.argv[1] if len(sys.argv) > 1 else None          # optional: ABS or PAIRED only
    if objective:
        s37.OBJECTIVES = {objective: s37.OBJECTIVES[objective]}
    sys.argv = [sys.argv[0], "--backbones", "MODERNTCN"]
    s37.main()

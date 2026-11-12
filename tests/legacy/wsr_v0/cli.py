"""Command line entry point:  python -m wsr.cli <command> --round <round> [...]"""
from __future__ import annotations

import argparse

from .common import ROUNDS, load_cfg


def main():
    ap = argparse.ArgumentParser(prog="wsr")
    ap.add_argument("--cfg", default="config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--round", required=True, choices=list(ROUNDS))
        return s

    p("pool", "write the intervention pool of a round").add_argument("--targets", nargs="+")
    m = p("measure", "measure base runs and interventions on the probe sets (1 GPU)")
    m.add_argument("--target", required=True)
    m.add_argument("--ids", nargs="*", help="default: every run of the pool not yet measured")
    p("labels", "refusal judge + item-level labels").add_argument("--targets", nargs="+")
    s = p("select", "apply the selection rule")
    s.add_argument("--targets", nargs="+")
    s.add_argument("--n-changed", dest="n_changed", type=int, default=30)
    s.add_argument("--n-same", dest="n_same", type=int, default=10)
    p("materialize", "write reader files and Text-Only descriptions").add_argument("--targets", nargs="+")
    se = p("session", "run one reader session (CPU)")
    se.add_argument("--reader", required=True)
    se.add_argument("--target", required=True)
    se.add_argument("--cond", required=True, choices=["weights", "text"])
    se.add_argument("--batch", type=int, required=True)
    se.add_argument("--framing", default="", choices=["", "self", "other"])
    se.add_argument("--hours", type=float, default=1.5)
    p("score", "score all sessions of a round")

    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    if a.cmd == "pool":
        from .select import cmd_pool as f
    elif a.cmd == "measure":
        from .measure import cmd_measure as f
    elif a.cmd == "labels":
        from .labels import cmd_labels as f
    elif a.cmd == "select":
        from .select import cmd_select as f
    elif a.cmd == "materialize":
        from .select import cmd_materialize as f
    elif a.cmd == "session":
        from .session import cmd_session as f
    else:
        from .score import cmd_score as f
    f(cfg, a)


if __name__ == "__main__":
    main()

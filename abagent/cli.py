"""Command line entry point.

    python -m abagent.cli demo                 # analyse every experiment, write reports/
    python -m abagent.cli analyze <exp_id>     # one experiment, print the memo
    python -m abagent.cli eval [--seeds 20]    # accuracy vs planted ground truth
"""

from __future__ import annotations

import argparse

from .agent import mock_policy, run_agent
from .config import REPORTS_DIR, build_context, load_data, load_registry
from .evaluate import evaluate, seed_sweep, write_report
from .llm import OFFLINE_BANNER, configured_model, get_llm, is_offline


def _banner() -> None:
    if is_offline():
        print(f"\n*** {OFFLINE_BANNER} ***\n")
    else:
        print(f"\nUsing model {configured_model()} via LiteLLM\n")


def cmd_demo(_: argparse.Namespace) -> None:
    df = load_data()
    print(f"{'experiment':<24}{'decision':<22}{'scope':<15}{'tools':>6}  validator")
    for exp_id in load_registry():
        r = run_agent(build_context(exp_id, df), llm=get_llm(mock_policy))
        status = "PASSED" if r.validation.passed else "FAILED"
        print(f"{exp_id:<24}{r.decision:<22}{r.scope:<15}{len(r.trace):>6}  {status}"
              + (f"   tokens={r.usage['prompt_tokens'] + r.usage['completion_tokens']:,} "
                 f"cost=${r.usage['estimated_cost_usd']:.4f}" if not is_offline() else ""))
    print(f"\nMemos and charts written to {REPORTS_DIR}/ (e.g. {REPORTS_DIR}/exp_discount_nudge.md)")


def cmd_analyze(args: argparse.Namespace) -> None:
    r = run_agent(build_context(args.experiment_id), llm=get_llm(mock_policy))
    print(r.plan + "\n" if r.plan else "")
    print(open(r.memo_path).read() if r.memo_path else r.memo_markdown)


def cmd_eval(args: argparse.Namespace) -> None:
    result = evaluate()
    sweep = seed_sweep(range(args.seeds)) if args.seeds else None
    path = write_report(result, sweep)
    print(path.read_text())
    print(f"Written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="abagent", description="A/B test readout agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="analyse every experiment").set_defaults(fn=cmd_demo)
    p = sub.add_parser("analyze", help="analyse one experiment")
    p.add_argument("experiment_id", choices=sorted(load_registry()))
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("eval", help="score decisions against planted ground truth")
    p.add_argument("--seeds", type=int, default=0,
                   help="also re-generate data with N seeds and report robustness")
    p.set_defaults(fn=cmd_eval)
    args = parser.parse_args()
    _banner()
    args.fn(args)


if __name__ == "__main__":
    main()

"""Paired comparison of two strategies on the same scenarios (no PyTorch needed).

    python rl/paired.py [--scenarios 500] [--hard] [--a rl_policy] [--b urgency_first]

Prints the mean difference a - b in score points with a 95% confidence interval (bootstrap over scenarios)
and how many scenarios a wins, ties and loses.
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import policy  # noqa: E402
from evaluate_hard import episode  # noqa: E402


def planner(name):
    return policy.plan if name == "rl_policy" else (lambda obs: agent.heuristic_plan(obs, name))


def compare(a, b, scenarios, hard, resamples=2000):
    pa, pb = planner(a), planner(b)
    diffs = [episode(s, pa, hard) - episode(s, pb, hard) for s in range(scenarios)]
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return {"mean": statistics.fmean(diffs), "low": means[int(0.025 * resamples)], "high": means[int(0.975 * resamples)],
            "wins": sum(d > 1e-9 for d in diffs), "ties": sum(abs(d) <= 1e-9 for d in diffs),
            "losses": sum(d < -1e-9 for d in diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--a", default="rl_policy")
    ap.add_argument("--b", default="urgency_first")
    args = ap.parse_args()
    r = compare(args.a, args.b, args.scenarios, args.hard)
    print(f"{args.a} - {args.b} on {args.scenarios} {'hard' if args.hard else 'normal'} scenarios: "
          f"{r['mean']:+.2f} points (95% CI {r['low']:+.2f} to {r['high']:+.2f}); "
          f"wins {r['wins']}, ties {r['ties']}, losses {r['losses']}")


if __name__ == "__main__":
    main()

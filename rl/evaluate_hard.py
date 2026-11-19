"""Compare the strategies on the harder simulator (reset(hard=True)); needs no PyTorch.

    python rl/evaluate_hard.py [--scenarios 500]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import policy  # noqa: E402
from env import ReliefEnv  # noqa: E402


def episode(seed, plan_fn, hard):
    env = ReliefEnv()
    obs = env.reset(seed=seed, chaos=True, hard=hard)
    while True:
        res = env.step({"assignments": plan_fn(obs)})
        obs = res["observation"]
        if res["done"]:
            st = env.state()
            return max(0.0, st["total_reward"] / st["max_reward"]) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=500)
    args = ap.parse_args()
    planners = {n: (lambda obs, n=n: agent.heuristic_plan(obs, n)) for n in ("urgency_first", "nearest_first", "fifo")}
    if policy.available():
        planners["rl_policy"] = policy.plan
    print(f"{'strategy':15s} {'normal':>8s} {'hard':>8s}   ({args.scenarios} scenarios each)")
    for name, fn in planners.items():
        row = [sum(episode(s, fn, hard) for s in range(args.scenarios)) / args.scenarios for hard in (False, True)]
        print(f"{name:15s} {row[0]:7.1f}% {row[1]:7.1f}%")


if __name__ == "__main__":
    main()

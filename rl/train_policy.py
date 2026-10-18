"""Train the dispatch policy with REINFORCE in the flood simulator.

    pip install -r requirements-train.txt
    python rl/train_policy.py --episodes 6000

Writes:
    policy_weights.json   weights used by the app (pure-Python inference, no PyTorch at serve time)
    rl/results.json       held-out evaluation against the rule-based strategies
    docs/rl_training.png  learning curve
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402
import policy  # noqa: E402
from env import ReliefEnv  # noqa: E402

TRAIN_SEED_BASE = 100_000  # training scenarios never overlap the evaluation seeds
EVAL_SEEDS = range(500)


class Scorer(nn.Module):
    """features -> logit; one tanh hidden layer (mirrored by policy.mlp)."""

    def __init__(self, n_in, hidden=16):
        super().__init__()
        self.l1, self.l2 = nn.Linear(n_in, hidden), nn.Linear(hidden, 1)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x))).squeeze(-1)

    def export(self):
        return {"W1": self.l1.weight.tolist(), "b1": self.l1.bias.tolist(),
                "W2": self.l2.weight[0].tolist(), "b2": self.l2.bias.item()}


def episode(seed, plan_fn):
    """Play one chaotic episode; returns normalised return (reward / best possible)."""
    env = ReliefEnv()
    obs = env.reset(seed=seed, chaos=True)
    while True:
        res = env.step({"assignments": plan_fn(obs)})
        obs = res["observation"]
        if res["done"]:
            st = env.state()
            return st["total_reward"] / st["max_reward"]


def policy_plan(need_net, depot_net, logps=None):
    """Sampling planner when logps is a list (training), greedy otherwise."""
    def choose(logits, kind):
        t = torch.stack(logits)
        if logps is None:
            return int(t.argmax())
        dist = torch.distributions.Categorical(logits=t)
        k = dist.sample()
        logps.append(dist.log_prob(k))
        return int(k)

    return lambda obs: policy.decide(obs, lambda f: need_net(torch.tensor(f)),
                                     lambda f: depot_net(torch.tensor(f)), choose)


def evaluate(plan_fn):
    scores = [max(0.0, episode(s, plan_fn)) * 100 for s in EVAL_SEEDS]
    return round(sum(scores) / len(scores), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    need_net, depot_net = Scorer(policy.N_NEED), Scorer(policy.N_PAIR)
    opt = torch.optim.Adam([*need_net.parameters(), *depot_net.parameters()], lr=args.lr)

    with torch.no_grad():
        untrained = evaluate(policy_plan(need_net, depot_net))
    print(f"untrained policy: {untrained}% on {len(EVAL_SEEDS)} held-out scenarios")

    curve, seed, t0 = [], TRAIN_SEED_BASE, time.time()
    for batch in range(args.episodes // args.batch):
        returns, logp_sums = [], []
        for _ in range(args.batch):
            logps = []
            returns.append(episode(seed, policy_plan(need_net, depot_net, logps)))
            logp_sums.append(torch.stack(logps).sum() if logps else torch.zeros(()))
            seed += 1
        r = torch.tensor(returns)
        adv = (r - r.mean()) / (r.std() + 1e-6)  # batch-mean baseline, normalised
        loss = -(adv * torch.stack(logp_sums)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(round(float(r.clamp(min=0).mean()) * 100, 2))
        if batch % 25 == 0:
            print(f"batch {batch:4d}  episodes {seed - TRAIN_SEED_BASE:6d}  mean score {curve[-1]:5.1f}%  "
                  f"{time.time() - t0:5.0f}s")

    with torch.no_grad():
        trained = evaluate(policy_plan(need_net, depot_net))
    weights = {"need": need_net.export(), "depot": depot_net.export(),
               "meta": {"algorithm": "REINFORCE", "episodes": seed - TRAIN_SEED_BASE, "batch": args.batch,
                        "hidden": 16, "eval_score": trained}}
    policy.WEIGHTS_PATH.write_text(json.dumps(weights), encoding="utf-8")

    # same evaluation for the rule-based strategies, and check the exported weights reproduce torch
    policy._weights = None
    results = {"eval_scenarios": len(EVAL_SEEDS), "untrained_policy": untrained, "trained_policy": trained,
               "trained_policy_pure_python": evaluate(policy.plan)}
    for name in ("urgency_first", "nearest_first", "fifo"):
        results[name] = evaluate(lambda obs, n=name: agent.heuristic_plan(obs, n))
    results["training_curve"] = curve
    (ROOT / "rl" / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "training_curve"}, indent=1))
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = results["training_curve"]
    k = 20
    smooth = [sum(curve[max(0, i - k + 1):i + 1]) / len(curve[max(0, i - k + 1):i + 1]) for i in range(len(curve))]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    ax.plot(curve, color="#b8c7da", lw=0.8, label="training batch (sampled actions)")
    ax.plot(smooth, color="#2f6f4f", lw=2, label="training, 20-batch average")
    for name, color in (("urgency_first", "#c27c0e"), ("nearest_first", "#6b7280"), ("fifo", "#b42318")):
        ax.axhline(results[name], color=color, ls="--", lw=1.2, label=f"{name} (held-out): {results[name]}%")
    ax.axhline(results["trained_policy"], color="#2f6f4f", ls=":", lw=1.5,
               label=f"trained policy (held-out, greedy): {results['trained_policy']}%")
    ax.set_xlabel("training batch")
    ax.set_ylabel("score, % of best possible")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("REINFORCE dispatch policy in the flood simulator", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "rl_training.png")


if __name__ == "__main__":
    main()

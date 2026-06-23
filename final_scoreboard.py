"""
Final scoreboard: heuristics vs the saved RecurrentPPO model in BOTH eval modes.

Evaluating PPO LSTM policies is subtle:
- ``deterministic=True``  -> argmax of policy logits.  Can collapse to a
  degenerate action when the policy is still high-entropy.
- ``deterministic=False`` -> sample from the policy.  Matches what the agent did
  during training; the fair measure of policy quality during/after training.

We report BOTH so the gap-vs-heuristic claim is unambiguous.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import student_simulator as sim
import evaluate_baselines as eb

from sb3_contrib import RecurrentPPO
from train_recurrent_ppo import RecurrentPolicyWrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runs/recurrent_ppo/best_model.zip")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=98_765)
    ap.add_argument("--sub-episode-length", type=int, default=20)
    ap.add_argument("--n-sub-episodes", type=int, default=4)
    args = ap.parse_args()

    students = sim.create_student_population(variants_per_ability=4, seed=0)

    policies = eb.default_heuristics()

    model_path = Path(args.model)
    if model_path.exists():
        model = RecurrentPPO.load(str(model_path))
        policies["RecurrentPPO[stochastic]"] = RecurrentPolicyWrapper(model, deterministic=False)
        policies["RecurrentPPO[argmax]"]    = RecurrentPolicyWrapper(model, deterministic=True)
    else:
        print(f"WARNING: model not found at {model_path}; running heuristics only.")

    print(f"\nFinal scoreboard  ({args.episodes} episodes, seed {args.seed}, "
          f"{args.n_sub_episodes}x{args.sub_episode_length} steps)\n")
    eb.print_scoreboard(
        policies, episodes=args.episodes, seed=args.seed,
        sub_episode_length=args.sub_episode_length, n_sub_episodes=args.n_sub_episodes,
        mastery_threshold=0.6, students=students,
    )
    print("\nGoal:  random << best_heuristic << trained  on TRUE_mastery_gain.\n")


if __name__ == "__main__":
    main()

"""
Baseline scoreboard for the topic-selection RL problem.

The whole point of this script is to measure the GAP between trivial policies and
the trained agent on the *same* environment and the *same* metrics.  A successful
RL result must satisfy:

        random  <<  best_heuristic  <<  trained

on TRUE latent mastery improvement (the thing we actually care about), not just on
the shaped env reward.  Absolute reward is farmable; the gap is what matters.

Policies compared
-----------------
* random            - uniform over currently-valid topics
* lowest_index      - always the lowest-index valid topic (a dumb fixed rule)
* prereq_heuristic  - among valid topics, pick the one with the highest *observed*
                      prerequisite readiness, preferring not-yet-covered topics
* drill_then_move   - stay on a topic until it looks handled, then move on
* trained:<path>    - a saved RL policy (a2c / double_dqn), greedy + masked

Run:  python evaluate_baselines.py --episodes 200 --model runs/a2c/models/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import rl_common as rc
from mcq_env import TopicSelectionMCQEnv, make_env

# obs layout (see TopicSelectionMCQEnv): 19 scalars then 6 blocks of n_topics each
_N_SCALAR = len(TopicSelectionMCQEnv.SCALAR_FEATURES)
_BLOCKS = TopicSelectionMCQEnv.TOPIC_BLOCKS


def _block(obs: np.ndarray, name: str, n_slots: int) -> np.ndarray:
    """Slice one per-slot feature block out of a flat observation vector.

    ``n_slots`` = the env's per-topic block length (now equal to the active
    chapter size, n_active_topics).
    """
    b = _BLOCKS.index(name)
    start = _N_SCALAR + b * n_slots
    return obs[start:start + n_slots]


# --------------------------------------------------------------------------- #
# policies: each is policy(env, obs, mask, rng) -> int (topic index)
# --------------------------------------------------------------------------- #
Policy = Callable[[TopicSelectionMCQEnv, np.ndarray, np.ndarray, np.random.Generator], int]


def _slot_from_topic(env, topic_idx: int) -> int | None:
    """Return the chapter slot for a global topic, or None if not in chapter."""
    hits = np.flatnonzero(env.active_idx == topic_idx)
    return int(hits[0]) if hits.size else None


def policy_random(env, obs, mask, rng) -> int:
    valid = np.flatnonzero(mask)
    return int(rng.choice(valid))


def policy_lowest_index(env, obs, mask, rng) -> int:
    return int(np.flatnonzero(mask)[0])


def policy_prereq_heuristic(env, obs, mask, rng) -> int:
    """Greedy on observed prerequisite readiness; break ties toward unseen slots."""
    n_slots = env._n_act
    readiness = _block(obs, "topic_prereq_ready_observed", n_slots)
    seen = _block(obs, "topic_seen_flag", n_slots)
    invalid = ~mask
    score = readiness.copy()
    score[seen > 0.5] -= 0.01        # tiny pull toward covering new ready slots
    score[invalid] = -np.inf
    return int(np.argmax(score))


def make_drill_then_move(mastered_acc: float = 0.75) -> Policy:
    """Stay on the previous slot until its observed accuracy looks high, then
    fall back to the prerequisite heuristic to pick the next slot."""
    def policy(env, obs, mask, rng) -> int:
        last_topic = env.last_topic_idx
        last_slot = _slot_from_topic(env, last_topic) if last_topic >= 0 else None
        if last_slot is not None and mask[last_slot]:
            attempts = env.topic_asked[last_topic]
            acc = env.topic_correct[last_topic] / max(attempts, 1.0)
            if attempts < 1 or acc < mastered_acc:
                return int(last_slot)
        return policy_prereq_heuristic(env, obs, mask, rng)
    return policy


def policy_style_oracle(env, obs, mask, rng) -> int:
    """CHEATING upper-bound: reads the hidden learning style + true mastery and
    plays the style-matched, prerequisite-respecting, spacing-aware move.

    This is NOT a deployable policy (it sees hidden state); it exists only to
    measure the CEILING.  If it far exceeds the blind heuristics, then inferring
    the hidden style is worth a lot -> there is real headroom for a recurrent RL
    agent to capture.  If it ties them, the dynamics need strengthening.
    """
    import curriculum as cur
    style = env.student.get("learning_style", "interleaved")
    mastery = env.student.get("topic_mastery", {})
    topics = env.topics
    valid = np.flatnonzero(mask)

    def true_readiness(i: int) -> float:
        prereqs = cur.prerequisites_of(topics[i])
        if not prereqs:
            return 1.0
        return float(np.mean([float(mastery.get(p, 0.0)) for p in prereqs]))

    last_topic_idx = env.last_topic_idx
    last_topic = topics[last_topic_idx] if last_topic_idx >= 0 else None
    last3 = [topics[j] for j in list(env.topic_selection_history)[-3:]]

    best_slot, best_score = int(valid[0]), -1e9
    for slot in valid:
        i = int(env.active_idx[slot])
        t = topics[i]
        ready = true_readiness(i)
        headroom = 1.0 - float(mastery.get(t, 0.0))
        # style match multiplier (mirrors the simulator's _style_multiplier)
        if style == "massed":
            sm = 1.80 if last_topic == t else 0.20
        elif style == "interleaved":
            sm = 0.20 if last_topic == t else (1.80 if t not in last3 else 0.70)
        else:  # blocked
            lc = cur.TOPIC_TO_CLUSTER.get(last_topic) if last_topic else None
            sm = 1.75 if (lc is not None and cur.TOPIC_TO_CLUSTER.get(t) == lc) else 0.25
        score = (ready ** 2) * sm * headroom    # expected immediate mastery gain
        if score > best_score:
            best_score, best_slot = score, int(slot)
    return best_slot


def make_trained(model_path: Path, device: torch.device) -> tuple[Policy, str]:
    ckpt = rc.load_model(model_path, device)
    algo = ckpt["algo"]
    model = rc.build_model_from_checkpoint(ckpt, device)

    def policy(env, obs, mask, rng) -> int:
        return rc.greedy_topic(model, algo, obs, mask, device)
    return policy, algo


# --------------------------------------------------------------------------- #
# rollout / aggregation
# --------------------------------------------------------------------------- #
def _topics_mastered(env: TopicSelectionMCQEnv, threshold: float) -> int:
    mastery = env.student.get("topic_mastery", {})
    return int(sum(1 for i in env.active_idx
                   if float(mastery.get(env.topics[i], 0.0)) >= threshold))


def run_policy(policy: Policy, *, episodes: int, seed: int, sub_episode_length: int,
               n_sub_episodes: int, mastery_threshold: float,
               students: list[dict[str, Any]] | None = None) -> dict[str, np.ndarray]:
    rewards, improvements, finals, accs, repeats, mastered = [], [], [], [], [], []
    reset_hook = getattr(policy, "reset", None)
    for ep in range(episodes):
        ep_seed = seed + ep
        env = make_env(students=students, sub_episode_length=sub_episode_length,
                       n_sub_episodes=n_sub_episodes, seed=ep_seed,
                       randomize_initial_ability=True)
        obs, _ = env.reset(seed=ep_seed)
        if callable(reset_hook):       # stateful (recurrent) policies reset per episode
            reset_hook()
        rng = np.random.default_rng(ep_seed)  # paired across policies for fair comparison
        done = False
        info: dict[str, Any] = {}
        n_repeat = steps = 0
        while not done:
            mask = env.valid_slot_mask().astype(bool)
            action = policy(env, obs, mask, rng)
            obs, _, term, trunc, info = env.step(action)
            n_repeat += int(bool(info.get("was_repeat", False)))
            steps += 1
            done = term or trunc
        rewards.append(float(info.get("total_agent_reward", 0.0)))
        improvements.append(float(info.get("mastery_improvement", 0.0)))
        finals.append(float(info.get("final_mean_mastery", 0.0)))
        accs.append(float(info.get("accuracy", 0.0)))
        repeats.append(n_repeat / max(1, steps))
        mastered.append(_topics_mastered(env, mastery_threshold))
    return {
        "reward": np.array(rewards),
        "mastery_gain": np.array(improvements),
        "final_mastery": np.array(finals),
        "accuracy": np.array(accs),
        "repeat_rate": np.array(repeats),
        "topics_mastered": np.array(mastered, dtype=float),
    }


def default_heuristics() -> dict[str, Policy]:
    """The fixed baseline policies, reused by the trainer's final scoreboard."""
    return {
        "random": policy_random,
        "lowest_index": policy_lowest_index,
        "prereq_heuristic": policy_prereq_heuristic,
        "drill_then_move": make_drill_then_move(),
        "style_oracle*": policy_style_oracle,   # cheating ceiling (sees hidden style)
    }


def _fmt(arr: np.ndarray) -> str:
    return f"{arr.mean():7.2f} +/- {arr.std():5.2f}"


def print_scoreboard(policies: dict[str, Policy], *, episodes: int, seed: int,
                     sub_episode_length: int, n_sub_episodes: int,
                     mastery_threshold: float, students=None) -> dict[str, dict[str, np.ndarray]]:
    header = (f"{'policy':18s} {'env_reward':>16s} {'TRUE_mastery_gain':>20s} "
              f"{'final_mastery':>16s} {'#mastered':>14s} {'accuracy':>14s} {'repeat':>14s}")
    print(header)
    print("-" * len(header))
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, pol in policies.items():
        res = run_policy(pol, episodes=episodes, seed=seed,
                         sub_episode_length=sub_episode_length,
                         n_sub_episodes=n_sub_episodes,
                         mastery_threshold=mastery_threshold, students=students)
        out[name] = res
        print(f"{name:18s} {_fmt(res['reward']):>16s} {_fmt(res['mastery_gain']):>20s} "
              f"{_fmt(res['final_mastery']):>16s} {_fmt(res['topics_mastered']):>14s} "
              f"{_fmt(res['accuracy']):>14s} {_fmt(res['repeat_rate']):>14s}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline scoreboard for topic-selection RL")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sub-episode-length", type=int, default=20)
    ap.add_argument("--n-sub-episodes", type=int, default=4)
    ap.add_argument("--mastery-threshold", type=float, default=0.6)
    ap.add_argument("--model", type=str, default=None,
                    help="path to a saved RL model (.pt) to include as 'trained'")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = rc.get_device(args.device)
    # one fixed student population shared by every policy -> apples to apples
    import student_simulator as sim
    students = sim.create_student_population(variants_per_ability=4, seed=args.seed)

    policies: dict[str, Policy] = default_heuristics()
    if args.model:
        pol, algo = make_trained(Path(args.model), device)
        policies[f"trained[{algo}]"] = pol

    print(f"\nScoreboard  ({args.episodes} episodes, seed {args.seed}, "
          f"{args.n_sub_episodes}x{args.sub_episode_length} steps, "
          f"mastery>= {args.mastery_threshold})\n")
    print_scoreboard(policies, episodes=args.episodes, seed=args.seed,
                     sub_episode_length=args.sub_episode_length,
                     n_sub_episodes=args.n_sub_episodes,
                     mastery_threshold=args.mastery_threshold, students=students)
    print("\nGoal:  random << best_heuristic << trained  on TRUE_mastery_gain.\n")


if __name__ == "__main__":
    main()

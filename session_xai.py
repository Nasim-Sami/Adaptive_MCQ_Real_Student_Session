"""
session_xai.py — Lightweight inline XAI for the real student session.

Explains, after every topic selection, WHY the RL model chose that topic:
  • Score of the chosen topic vs. the runner-up (and margin)
  • Top-3 scoring available topics
  • Which feature groups drove the decision (fast occlusion, no gradients)
  • Curriculum consequence: prereqs, transfer boosts, difficulty target

Designed to run fast enough for an interactive session (<0.1 s per call on CPU).
Does NOT compute gradient saliency (too slow interactively).

Supports all five trained-model formats: legacy memoryless ``a2c`` /
``double_dqn`` (rl_common.ActorCritic / QNetwork), the from-scratch recurrent
``a2c_lstm`` / ``double_dqn_lstm`` (rl_common.RecurrentActorCritic /
RecurrentQNetwork, carrying an (h, c) hidden state), and ``recurrent_ppo``
(sb3-contrib RecurrentPPO, via its policy's ``get_distribution``).

Action-space note: since the v2.4 chapter-focus change, every model's output
is a CHAPTER SLOT index (0..n_active_topics-1), not a global topic index -
every lookup into ``env.topics`` here goes through ``env.active_idx[slot]``.

Hidden-state note (recurrent algos only): the explanation must use the SAME
hidden state that produced the actual decision being explained - the caller
must pass the hidden state as it was BEFORE that decision (not after), and
this module never mutates or advances it; occlusion counterfactuals reuse it
unchanged so the comparison is fair.

Public API
----------
    from session_xai import explain_topic_choice, format_xai_block

    xai = explain_topic_choice(model, algo, obs, mask, env, device,
                               selected_slot=slot, hidden_state=hidden_before)
    print(format_xai_block(xai))
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import curriculum
import rl_common as rc
from mcq_env import ability_to_target_difficulty

RECURRENT_TORCH_ALGOS = ("a2c_lstm", "double_dqn_lstm")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _get_scores(model, algo: str, obs: np.ndarray, mask: np.ndarray, device,
                hidden_state=None) -> np.ndarray:
    """Return per-CHAPTER-SLOT scores (Q-values or action probabilities).

    ``hidden_state`` is required for recurrent algos (a2c_lstm,
    double_dqn_lstm: an (h, c) torch tensor tuple from rl_common's
    init_hidden/forward; recurrent_ppo: an (h, c) numpy tuple as returned by
    sb3's ``model.predict`` / RecurrentPolicyWrapper, or None for a fresh
    episode start).
    """
    mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)

    if algo == "a2c":
        with torch.no_grad():
            logits, _ = model(rc.tensor(obs[None, :], device))
            masked = logits.masked_fill(~mask_t, -1e9)
            scores = F.softmax(masked, dim=-1)[0].cpu().numpy()

    elif algo == "a2c_lstm":
        with torch.no_grad():
            obs_t = rc.tensor(obs[None, None, :], device)
            logits, _, _ = model(obs_t, hidden_state)
            masked = logits[0, 0].masked_fill(~mask_t[0], -1e9)
            scores = F.softmax(masked, dim=-1).cpu().numpy()

    elif algo == "double_dqn_lstm":
        with torch.no_grad():
            obs_t = rc.tensor(obs[None, None, :], device)
            q, _ = model(obs_t, hidden_state)
            scores = q[0, 0].masked_fill(~mask_t[0], -1e9).cpu().numpy()

    elif algo == "recurrent_ppo":
        obs_t = rc.tensor(obs[None, :], device)
        shape = model.policy.lstm_hidden_state_shape
        if hidden_state is None:
            h = torch.zeros(*shape, device=device)
            c = torch.zeros(*shape, device=device)
            episode_starts = torch.ones(1, device=device)
        else:
            h = torch.as_tensor(hidden_state[0], dtype=torch.float32, device=device)
            c = torch.as_tensor(hidden_state[1], dtype=torch.float32, device=device)
            episode_starts = torch.zeros(1, device=device)
        with torch.no_grad():
            dist, _ = model.policy.get_distribution(obs_t, (h, c), episode_starts)
            probs = dist.distribution.probs[0].cpu().numpy()
        scores = np.where(mask, probs, -1e9)

    else:  # "dqn" / "double_dqn" (legacy memoryless QNetwork)
        with torch.no_grad():
            q = model(rc.tensor(obs[None, :], device)).masked_fill(~mask_t, -1e9)
            scores = q[0].cpu().numpy()

    return scores


# ---------------------------------------------------------------------------
# Fast occlusion attribution (only 6 groups, very fast)
# ---------------------------------------------------------------------------
_GROUP_LABELS = {
    "recent_performance":              "Recent perf.",
    "observed_topic_accuracy":         "Obs. accuracy",
    "observed_topic_wrong_rate":       "Obs. wrong rate",
    "observed_topic_asked_rate":       "Obs. asked rate",
    "observed_topic_seen":             "Topic seen flag",
    "observed_prerequisite_readiness": "Prereq readiness",
    "topic_availability":              "Availability",
}


def _feature_groups(env) -> dict[str, list[int]]:
    """Feature-block index ranges. Per-topic blocks are CHAPTER-SLOT-length
    (env._n_act), matching the obs layout since the v2.4 chapter focus."""
    groups: dict[str, list[int]] = {"recent_performance": list(range(env.n_scalar))}
    start = env.n_scalar
    n_slots = env._n_act
    pretty = {
        "topic_asked_rate":         "observed_topic_asked_rate",
        "topic_accuracy":           "observed_topic_accuracy",
        "topic_wrong_rate":         "observed_topic_wrong_rate",
        "topic_seen_flag":          "observed_topic_seen",
        "topic_prereq_ready_observed": "observed_prerequisite_readiness",
        "topic_available_flag":     "topic_availability",
    }
    for block in env.TOPIC_BLOCKS:
        if block not in pretty:
            start += n_slots
            continue
        groups[pretty[block]] = list(range(start, start + n_slots))
        start += n_slots
    return groups


def _fast_occlusion(model, algo: str, obs: np.ndarray, mask: np.ndarray,
                    selected: int, base_score: float, env, device,
                    hidden_state=None) -> list[dict]:
    attribution = []
    for name, idx in _feature_groups(env).items():
        obs2 = obs.copy()
        obs2[idx] = 0.0
        scores2 = _get_scores(model, algo, obs2, mask, device, hidden_state)
        drop = base_score - float(scores2[selected])
        new_choice = int(np.argmax(np.where(mask, scores2, -1e18)))
        attribution.append({
            "group": name,
            "label": _GROUP_LABELS.get(name, name),
            "support": round(drop, 5),
            "flips": new_choice != selected,
            "alt_if_removed": env.topics[int(env.active_idx[new_choice])] if new_choice != selected else None,
        })
    attribution.sort(key=lambda d: abs(d["support"]), reverse=True)
    return attribution


# ---------------------------------------------------------------------------
# Main explain function
# ---------------------------------------------------------------------------
def explain_topic_choice(
    model,
    algo: str,
    obs: np.ndarray,
    mask: np.ndarray,
    env,
    device,
    selected_slot: int,
    hidden_state=None,
    precomputed_target_diff: float | None = None,
) -> dict[str, Any]:
    """
    Explain the RL model's current topic selection.

    Parameters
    ----------
    selected_slot : The CHAPTER SLOT actually chosen for this step (the same
        value used to pick the question) - explains THAT choice, rather than
        independently recomputing an argmax that could differ for stochastic
        policies (e.g. RecurrentPPO is sampled, not greedy, by default).
    hidden_state : The recurrent hidden state as it was BEFORE this decision
        (required for a2c_lstm / double_dqn_lstm / recurrent_ppo; ignored for
        the legacy memoryless algos). Not mutated.
    precomputed_target_diff : If provided (from compute_target_difficulty),
        this richer value is shown in the XAI box instead of the simple
        ability_to_target_difficulty fallback.

    Returns a dict with:
        selected_topic, score_kind, selected_score, runner_up,
        runner_up_score, margin, top3, attribution,
        curriculum (prereqs, transfer, difficulty target)
    """
    scores = _get_scores(model, algo, obs, mask, device, hidden_state)
    score_kind = "action_prob" if algo in ("a2c", "a2c_lstm", "recurrent_ppo") else "q_value"

    # Rank available slots by score
    ranked_idx = np.argsort(np.where(mask, scores, -1e18))[::-1]
    top_ranked = int(ranked_idx[0])

    # Explain the ACTUAL choice, not necessarily the top-ranked one (a
    # stochastic policy like RecurrentPPO samples rather than argmaxes).
    selected = int(selected_slot)
    was_top_ranked = (selected == top_ranked)
    selected_topic = env.topics[int(env.active_idx[selected])]
    selected_score = float(scores[selected])

    # Runner-up = best-scoring slot that isn't the one actually chosen.
    runner_up_idx = next((int(i) for i in ranked_idx if int(i) != selected), selected)
    runner_up = env.topics[int(env.active_idx[runner_up_idx])]
    runner_up_score = float(scores[runner_up_idx])
    margin = selected_score - runner_up_score

    top3 = [
        {
            "rank": r + 1,
            "topic": env.topics[int(env.active_idx[i])],
            "score": round(float(scores[i]), 5),
        }
        for r, i in enumerate(ranked_idx[:3])
    ]

    # Fast occlusion (evaluated at the SAME hidden state as the real decision)
    attribution = _fast_occlusion(model, algo, obs, mask, selected, selected_score,
                                  env, device, hidden_state)

    # Curriculum consequence
    simple_target = ability_to_target_difficulty(env.effective_ability)
    target_diff = precomputed_target_diff if precomputed_target_diff is not None else simple_target
    target_diff_is_rich = precomputed_target_diff is not None
    prereqs = curriculum.prerequisites_of(selected_topic)
    transfers = curriculum.transfer_targets(selected_topic)
    boosts = {t: round(w, 3) for t, w in transfers.items() if w > 0}
    interferes = {t: round(w, 3) for t, w in transfers.items() if w < 0}

    # Observed prereq readiness for the chosen topic (full-obs-length array,
    # indexed by global topic index, NOT slot)
    obs_prereq_ready = float(env._observed_prereq_ready()[int(env.active_idx[selected])])

    return {
        "selected_topic": selected_topic,
        "score_kind": score_kind,
        "selected_score": round(selected_score, 5),
        "was_top_ranked": was_top_ranked,
        "runner_up": runner_up,
        "runner_up_score": round(runner_up_score, 5),
        "margin": round(margin, 5),
        "top3": top3,
        "attribution": attribution[:4],   # top-4 groups for display
        "curriculum": {
            "target_difficulty": round(target_diff, 2),
            "simple_target_difficulty": round(simple_target, 2),
            "target_is_rich": target_diff_is_rich,
            "effective_ability": round(env.effective_ability, 2),
            "observed_prereq_readiness": round(obs_prereq_ready, 3),
            "direct_prerequisites": prereqs,
            "will_boost": boosts,
            "may_interfere": interferes,
        },
    }


# ---------------------------------------------------------------------------
# Terminal formatter
# ---------------------------------------------------------------------------
def format_xai_block(xai: dict[str, Any], step: int) -> str:
    """Return a compact multi-line string for terminal display."""
    lines: list[str] = []
    W = 72

    lines.append("┌" + "─" * (W - 2) + "┐")
    lines.append(f"│  XAI — Step {step+1:02d}: Why the model chose this topic" + " " * (W - 46 - len(str(step+1))) + "│")
    lines.append("├" + "─" * (W - 2) + "┤")

    # Selected vs runner-up
    sk = xai["score_kind"]
    sel = xai["selected_topic"]
    sel_s = xai["selected_score"]
    rup = xai["runner_up"]
    rup_s = xai["runner_up_score"]
    margin = xai["margin"]

    lines.append(f"│  ► Chosen   : {sel:<40} {sk}={sel_s:+.4f}  │")
    if not xai.get("was_top_ranked", True):
        lines.append("│    (note: sampled stochastically - not the top-ranked option)".ljust(W - 1) + "│")
    lines.append(f"│  ○ Runner-up: {rup:<40} {sk}={rup_s:+.4f}  │")
    lines.append(f"│    Margin   : {margin:+.5f}  ({'HIGH' if abs(margin) > 0.05 else 'low'} confidence)  " + " " * max(0, W - 53) + "│")

    # Top 3
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  Top-3 available topics:".ljust(W - 1) + "│")
    for t in xai["top3"]:
        marker = "►" if t["rank"] == 1 else " "
        lines.append(f"│    {marker} {t['rank']}. {t['topic']:<42} {t['score']:+.4f}  │")

    # Attribution
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  Evidence that drove the choice (occlusion):".ljust(W - 1) + "│")
    for a in xai["attribution"]:
        direction = "▲ supports" if a["support"] > 0 else "▼ opposes "
        flip_tag = " ← FLIPS to: " + (a["alt_if_removed"] or "?") if a["flips"] else ""
        label = a["label"][:22]
        lines.append(f"│    {label:<22} {direction} {a['support']:+.4f}{flip_tag}".ljust(W - 1) + "│")

    # Curriculum
    c = xai["curriculum"]
    lines.append("├" + "─" * (W - 2) + "┤")
    lines.append("│  Curriculum consequence:".ljust(W - 1) + "│")
    if c.get("target_is_rich"):
        diff_str = (f"target={c['target_difficulty']:.2f} (rich: ability+perf+streaks)  "
                    f"simple={c['simple_target_difficulty']:.2f}")
    else:
        diff_str = f"target difficulty={c['target_difficulty']:.2f}"
    lines.append(f"│    Ability={c['effective_ability']:.2f}  →  {diff_str}".ljust(W - 1)[:W - 1] + "│")
    lines.append(f"│    Prereq readiness={c['observed_prereq_readiness']:.2f}".ljust(W - 1) + "│")
    if c["direct_prerequisites"]:
        prereq_str = ", ".join(c["direct_prerequisites"])
        lines.append(f"│    Prerequisites : {prereq_str}".ljust(W - 1)[:W - 1] + "│")
    if c["will_boost"]:
        boost_str = ", ".join(f"{t} (+{w})" for t, w in list(c["will_boost"].items())[:3])
        lines.append(f"│    Transfer boost: {boost_str}".ljust(W - 1)[:W - 1] + "│")
    if c["may_interfere"]:
        int_str = ", ".join(f"{t} ({w})" for t, w in list(c["may_interfere"].items())[:2])
        lines.append(f"│    Interference  : {int_str}".ljust(W - 1)[:W - 1] + "│")

    lines.append("└" + "─" * (W - 2) + "┘")
    return "\n".join(lines)

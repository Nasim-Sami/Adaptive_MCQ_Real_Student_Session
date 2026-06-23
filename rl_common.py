"""
Shared RL infrastructure for the June-15 topic-selection system.

This module holds the pieces that the two *separate* algorithm scripts
(``train_double_dqn.py`` and ``train_a2c.py``) have in common: the neural
networks, the experience-replay buffer, action masking over topics, evaluation,
checkpoint I/O and CSV logging.  The learning *algorithms* themselves live
entirely inside their own scripts - this file deliberately contains no Double
DQN / A2C update logic, so each algorithm remains a self-contained, readable
script (and we never call stable-baselines).
"""
from __future__ import annotations

import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from mcq_env import TopicSelectionMCQEnv, make_env
import student_simulator as sim


# ---------------------------------------------------------------------------
# reproducibility / device
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        return self.actor(features), self.critic(features).squeeze(-1)


# ---------------------------------------------------------------------------
# recurrent (LSTM) networks - memory-equipped A2C / Double DQN
#
# Both take a (batch, seq_len, obs_dim) sequence (use seq_len=1 for single-step
# inference) and an optional (h, c) hidden state, returning the updated hidden
# state alongside the usual outputs so callers can carry memory across an
# episode (acting) or unroll a full stored episode (training).
# ---------------------------------------------------------------------------
class RecurrentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, lstm_hidden: int = 256) -> None:
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.pre = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_actions)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        x = self.pre(obs_seq)
        out, new_hidden = self.lstm(x, hidden)
        return self.head(out), new_hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, lstm_hidden: int = 256) -> None:
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.pre = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        self.actor = nn.Linear(lstm_hidden, n_actions)
        self.critic = nn.Linear(lstm_hidden, 1)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        x = self.pre(obs_seq)
        out, new_hidden = self.lstm(x, hidden)
        return self.actor(out), self.critic(out).squeeze(-1), new_hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)


# ---------------------------------------------------------------------------
# topic action masking
#
# NOTE: the env's action space is Discrete(n_active_topics) - a CHAPTER SLOT
# index, not a global topic index (see mcq_env.TopicSelectionMCQEnv.step).
# This mask must therefore be slot-length (env.valid_slot_mask()), matching
# what the policy networks output. Map slot -> global topic via
# ``env.active_idx[slot]`` wherever a topic name/index is needed downstream.
# ---------------------------------------------------------------------------
def valid_topic_mask(env: TopicSelectionMCQEnv) -> np.ndarray:
    return env.valid_slot_mask().astype(bool)


def masked_argmax(values: torch.Tensor, mask: np.ndarray) -> int:
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
    return int(torch.argmax(values.masked_fill(~mask_t, -1e9)).item())


# ---------------------------------------------------------------------------
# replay buffer (DQN)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int) -> None:
        self.capacity = capacity
        self.n_actions = n_actions
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, done, next_mask) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.next_masks[self.pos] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, len(self), size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "next_masks": self.next_masks[idx],
        }

    def state_dict(self) -> dict[str, Any]:
        return {"obs": self.obs, "next_obs": self.next_obs, "actions": self.actions,
                "rewards": self.rewards, "dones": self.dones, "next_masks": self.next_masks,
                "pos": self.pos, "full": self.full}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.obs[:] = state["obs"]; self.next_obs[:] = state["next_obs"]
        self.actions[:] = state["actions"]; self.rewards[:] = state["rewards"]
        self.dones[:] = state["dones"]; self.next_masks[:] = state["next_masks"]
        self.pos = int(state["pos"]); self.full = bool(state["full"])


# ---------------------------------------------------------------------------
# episode replay buffer (Recurrent / DRQN-style Double DQN)
#
# The env's episodes are fixed-length (sub_episode_length * n_sub_episodes,
# always terminating exactly there - see TopicSelectionMCQEnv.step), so a
# whole episode can be stored as one fixed-size row with no padding/burn-in
# bookkeeping. Each row is later unrolled through the LSTM as one sequence.
# ---------------------------------------------------------------------------
class EpisodeReplayBuffer:
    def __init__(self, capacity_episodes: int, episode_len: int, obs_dim: int, n_actions: int) -> None:
        self.capacity = capacity_episodes
        self.episode_len = episode_len
        self.n_actions = n_actions
        self.obs = np.zeros((capacity_episodes, episode_len, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity_episodes, episode_len, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity_episodes, episode_len), dtype=np.int64)
        self.rewards = np.zeros((capacity_episodes, episode_len), dtype=np.float32)
        self.dones = np.zeros((capacity_episodes, episode_len), dtype=np.float32)
        self.next_masks = np.zeros((capacity_episodes, episode_len, n_actions), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add_episode(self, obs, actions, rewards, next_obs, dones, next_masks) -> None:
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = actions
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.next_masks[i] = next_masks
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, len(self), size=batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "next_masks": self.next_masks[idx],
        }

    def state_dict(self) -> dict[str, Any]:
        return {"obs": self.obs, "next_obs": self.next_obs, "actions": self.actions,
                "rewards": self.rewards, "dones": self.dones, "next_masks": self.next_masks,
                "pos": self.pos, "full": self.full}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.obs[:] = state["obs"]; self.next_obs[:] = state["next_obs"]
        self.actions[:] = state["actions"]; self.rewards[:] = state["rewards"]
        self.dones[:] = state["dones"]; self.next_masks[:] = state["next_masks"]
        self.pos = int(state["pos"]); self.full = bool(state["full"])


# ---------------------------------------------------------------------------
# logging / checkpoint helpers
# ---------------------------------------------------------------------------
def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_dirs(save_dir: Path) -> dict[str, Path]:
    paths = {
        "root": save_dir,
        "models": save_dir / "models",
        "checkpoints": save_dir / "checkpoints",
        "logs": save_dir / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


RECURRENT_ALGOS = ("a2c_lstm", "double_dqn_lstm")


def save_model(path: Path, *, algo: str, model: nn.Module, obs_dim: int, n_actions: int,
               hidden: int, lstm_hidden: int = 0, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": algo,
        "model_state": model.state_dict(),
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "hidden": hidden,
        "lstm_hidden": lstm_hidden,
        "topics": list(sim.curriculum.TOPICS),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


# ---- full training-state checkpoints (for resuming training) --------------
def save_train_checkpoint(path: Path, *, algo: str, step: int, episode: int, best_eval: float,
                          model: nn.Module, optimizer, obs_dim: int, n_actions: int, hidden: int,
                          lstm_hidden: int = 0,
                          replay: "ReplayBuffer | EpisodeReplayBuffer | None" = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": algo, "step": step, "episode": episode, "best_eval": best_eval,
        "obs_dim": obs_dim, "n_actions": n_actions, "hidden": hidden, "lstm_hidden": lstm_hidden,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "topics": list(sim.curriculum.TOPICS),
    }
    if replay is not None:
        payload["replay_state"] = replay.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_train_checkpoint(path: Path, *, model: nn.Module, optimizer, device: torch.device,
                          replay: "ReplayBuffer | EpisodeReplayBuffer | None" = None) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if replay is not None and "replay_state" in ckpt:
        replay.load_state_dict(ckpt["replay_state"])
    return int(ckpt.get("step", 0)), int(ckpt.get("episode", 0)), float(ckpt.get("best_eval", -1e18))


def build_model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> nn.Module:
    algo = ckpt["algo"]
    obs_dim, n_actions, hidden = ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"]
    if algo == "a2c":
        model: nn.Module = ActorCritic(obs_dim, n_actions, hidden)
    elif algo == "a2c_lstm":
        model = RecurrentActorCritic(obs_dim, n_actions, hidden, int(ckpt["lstm_hidden"]))
    elif algo == "double_dqn_lstm":
        model = RecurrentQNetwork(obs_dim, n_actions, hidden, int(ckpt["lstm_hidden"]))
    else:
        model = QNetwork(obs_dim, n_actions, hidden)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# greedy action helpers (used in eval, real session, XAI)
# ---------------------------------------------------------------------------
def greedy_topic_from_q(net: QNetwork, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        q = net(tensor(obs[None, :], device))[0]
    return masked_argmax(q, mask)


def greedy_topic_from_actor(net: ActorCritic, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        logits, _ = net(tensor(obs[None, :], device))
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        logits = logits.masked_fill(~mask_t, -1e9)
    return int(torch.argmax(logits, dim=-1).item())


def greedy_topic(model: nn.Module, algo: str, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    if algo == "a2c":
        return greedy_topic_from_actor(model, obs, mask, device)  # type: ignore[arg-type]
    return greedy_topic_from_q(model, obs, mask, device)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# recurrent single-step inference (acting) - carries (h, c) across an episode
# ---------------------------------------------------------------------------
def greedy_topic_from_recurrent_q(net: RecurrentQNetwork, obs: np.ndarray, mask: np.ndarray,
                                  hidden, device: torch.device) -> tuple[int, Any]:
    with torch.no_grad():
        obs_t = tensor(obs[None, None, :], device)
        q, new_hidden = net(obs_t, hidden)
    return masked_argmax(q[0, 0], mask), new_hidden


def greedy_topic_from_recurrent_actor(net: RecurrentActorCritic, obs: np.ndarray, mask: np.ndarray,
                                      hidden, device: torch.device) -> tuple[int, Any]:
    with torch.no_grad():
        obs_t = tensor(obs[None, None, :], device)
        logits, _, new_hidden = net(obs_t, hidden)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
        logits = logits[0, 0].masked_fill(~mask_t, -1e9)
    return int(torch.argmax(logits).item()), new_hidden


class RecurrentTorchPolicyWrapper:
    """Adapts a from-scratch recurrent net (RecurrentActorCritic / RecurrentQNetwork)
    to the (env, obs, mask, rng) -> action API used by evaluate_baselines /
    final_scoreboard, carrying the LSTM hidden state across one episode.

    Mirrors train_recurrent_ppo.RecurrentPolicyWrapper (the sb3-contrib
    equivalent) so all three trained agents can be scored with the same
    harness. Returns a CHAPTER-SLOT index, same as env.action_space expects.
    """

    def __init__(self, model: nn.Module, algo: str, device: torch.device) -> None:
        self.model = model
        self.algo = algo
        self.device = device
        self.hidden = None

    def reset(self) -> None:
        self.hidden = None

    def __call__(self, env, obs: np.ndarray, mask: np.ndarray, rng) -> int:
        if self.algo == "a2c_lstm":
            action, self.hidden = greedy_topic_from_recurrent_actor(self.model, obs, mask, self.hidden, self.device)
        else:
            action, self.hidden = greedy_topic_from_recurrent_q(self.model, obs, mask, self.hidden, self.device)
        return action


# ---------------------------------------------------------------------------
# evaluation (greedy) - reports learning-centric metrics
# ---------------------------------------------------------------------------
EVAL_FIELDS = [
    "run_id", "timestep", "algo",
    "eval_mean_agent_reward", "eval_mean_accuracy",
    "eval_mean_mastery_improvement", "eval_mean_final_mastery",
    "eval_mean_learning_gain", "eval_repeat_rate",
]


def evaluate(model: nn.Module, algo: str, *, n_episodes: int, sub_episode_length: int,
             n_sub_episodes: int, seed: int, device: torch.device,
             students: list[dict[str, Any]] | None = None) -> dict[str, float]:
    rewards, accs, improvements, finals, gains, repeats = [], [], [], [], [], []
    for ep in range(n_episodes):
        env = make_env(students=students, sub_episode_length=sub_episode_length,
                       n_sub_episodes=n_sub_episodes, seed=seed + ep, randomize_initial_ability=True)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info: dict[str, Any] = {}
        n_repeat = 0
        steps = 0
        while not done:
            mask = valid_topic_mask(env)
            action = greedy_topic(model, algo, obs, mask, device)
            obs, _, term, trunc, info = env.step(action)
            n_repeat += int(bool(info.get("was_repeat", False)))
            steps += 1
            done = term or trunc
        rewards.append(float(info.get("total_agent_reward", 0.0)))
        accs.append(float(info.get("accuracy", 0.0)))
        improvements.append(float(info.get("mastery_improvement", 0.0)))
        finals.append(float(info.get("final_mean_mastery", 0.0)))
        gains.append(float(info.get("total_learning_gain", 0.0)))
        repeats.append(n_repeat / max(1, steps))
    return {
        "eval_mean_agent_reward": float(np.mean(rewards)),
        "eval_mean_accuracy": float(np.mean(accs)),
        "eval_mean_mastery_improvement": float(np.mean(improvements)),
        "eval_mean_final_mastery": float(np.mean(finals)),
        "eval_mean_learning_gain": float(np.mean(gains)),
        "eval_repeat_rate": float(np.mean(repeats)),
    }


def evaluate_recurrent(model: nn.Module, algo: str, *, n_episodes: int, sub_episode_length: int,
                       n_sub_episodes: int, seed: int, device: torch.device,
                       students: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Same as ``evaluate`` but for RecurrentActorCritic / RecurrentQNetwork models -
    resets the LSTM hidden state to zero at the start of every episode."""
    policy = RecurrentTorchPolicyWrapper(model, algo, device)
    rewards, accs, improvements, finals, gains, repeats = [], [], [], [], [], []
    for ep in range(n_episodes):
        env = make_env(students=students, sub_episode_length=sub_episode_length,
                       n_sub_episodes=n_sub_episodes, seed=seed + ep, randomize_initial_ability=True)
        obs, _ = env.reset(seed=seed + ep)
        policy.reset()
        done = False
        info: dict[str, Any] = {}
        n_repeat = 0
        steps = 0
        while not done:
            mask = valid_topic_mask(env)
            action = policy(env, obs, mask, None)
            obs, _, term, trunc, info = env.step(action)
            n_repeat += int(bool(info.get("was_repeat", False)))
            steps += 1
            done = term or trunc
        rewards.append(float(info.get("total_agent_reward", 0.0)))
        accs.append(float(info.get("accuracy", 0.0)))
        improvements.append(float(info.get("mastery_improvement", 0.0)))
        finals.append(float(info.get("final_mean_mastery", 0.0)))
        gains.append(float(info.get("total_learning_gain", 0.0)))
        repeats.append(n_repeat / max(1, steps))
    return {
        "eval_mean_agent_reward": float(np.mean(rewards)),
        "eval_mean_accuracy": float(np.mean(accs)),
        "eval_mean_mastery_improvement": float(np.mean(improvements)),
        "eval_mean_final_mastery": float(np.mean(finals)),
        "eval_mean_learning_gain": float(np.mean(gains)),
        "eval_repeat_rate": float(np.mean(repeats)),
    }


TRAIN_EPISODE_FIELDS = [
    "run_id", "timestep", "episode", "algo",
    "student_id", "hidden_learning_style", "hidden_initial_effective_ability",
    "total_agent_reward", "total_student_reward", "accuracy",
    "initial_mean_mastery", "final_mean_mastery", "mastery_improvement",
    "total_learning_gain", "topic_sequence",
]


def summarize_training_episode(info: dict[str, Any], reset_info: dict[str, Any]) -> dict[str, Any]:
    rows = info.get("episode_rows", [])
    return {
        "student_id": reset_info.get("student_id", ""),
        "hidden_learning_style": reset_info.get("hidden_learning_style", ""),
        "hidden_initial_effective_ability": reset_info.get("hidden_initial_effective_ability", ""),
        "total_agent_reward": info.get("total_agent_reward", ""),
        "total_student_reward": info.get("total_student_reward", ""),
        "accuracy": info.get("accuracy", ""),
        "initial_mean_mastery": info.get("initial_mean_mastery", ""),
        "final_mean_mastery": info.get("final_mean_mastery", ""),
        "mastery_improvement": info.get("mastery_improvement", ""),
        "total_learning_gain": info.get("total_learning_gain", ""),
        "topic_sequence": " | ".join(str(r.get("selected_topic", "")) for r in rows),
    }


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return end
    return start + min(step / duration, 1.0) * (end - start)

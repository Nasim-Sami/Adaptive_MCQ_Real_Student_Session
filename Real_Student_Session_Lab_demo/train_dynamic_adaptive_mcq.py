from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from dynamic_ability_env import DynamicAbilityAdaptiveMCQEnv
from dynamic_student_simulator import (
    MAX_ABILITY,
    MIN_ABILITY,
    QUESTIONS_WITH_METADATA,
    create_student,
    create_student_population,
)


EVAL_ABILITIES = [10, 15, 20, 25, 30]

EVAL_FIELDS = [
    "run_id",
    "timestep",
    "algo",
    "eval_mean_agent_reward",
    "eval_mean_student_reward",
    "eval_mean_accuracy",
    "eval_mean_time_taken",
    "eval_mean_time_ratio",
    "eval_invalid_action_rate",
    "eval_effective_ability_start",
    "eval_effective_ability_end",
    "eval_effective_ability_delta",
    "eval_episode_length",
]

TRAIN_EPISODE_FIELDS = [
    "run_id",
    "timestep",
    "episode",
    "student_id",
    "initial_student_ability",
    "initial_effective_ability",
    "final_effective_ability",
    "total_agent_reward",
    "total_student_reward",
    "accuracy",
    "total_time_taken",
    "invalid_action_count",
    "question_sequence",
    "inherent_difficulty_sequence",
    "target_difficulty_sequence",
    "difficulty_gap_sequence",
    "difficulty_reward_sequence",
    "topic_reward_sequence",
    "diversity_reward_sequence",
    "response_label_sequence",
    "acting_mode_sequence",
    "acting_ability_sequence",
    "suggested_learning_path",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_dirs(save_dir: Path) -> dict[str, Path]:
    paths = {
        "root": save_dir,
        "models": save_dir / "models",
        "checkpoints": save_dir / "checkpoints",
        "logs": save_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def make_env(students: list[dict[str, Any]], episode_length: int, seed: int) -> DynamicAbilityAdaptiveMCQEnv:
    env = DynamicAbilityAdaptiveMCQEnv(
        questions=QUESTIONS_WITH_METADATA,
        students=students,
        episode_length=episode_length,
        seed=seed,
        repair_invalid_action=True,
        random_first_question=False,
    )
    env.action_space.seed(seed)
    return env


def valid_action_mask(env: DynamicAbilityAdaptiveMCQEnv) -> np.ndarray:
    mask = env.asked_mask < 0.5
    if not np.any(mask):
        mask = np.ones(env.n_questions, dtype=bool)
    return mask.astype(bool)


def valid_action_mask_from_obs(obs: np.ndarray, n_questions: int) -> np.ndarray:
    asked_start = 25 + 10 + n_questions
    asked_end = asked_start + n_questions
    asked_mask = obs[asked_start:asked_end]
    mask = asked_mask < 0.5
    if not np.any(mask):
        mask = np.ones(n_questions, dtype=bool)
    return mask.astype(bool)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = float(done)
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
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "obs": self.obs,
            "next_obs": self.next_obs,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "pos": self.pos,
            "full": self.full,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.obs[:] = state["obs"]
        self.next_obs[:] = state["next_obs"]
        self.actions[:] = state["actions"]
        self.rewards[:] = state["rewards"]
        self.dones[:] = state["dones"]
        self.pos = int(state["pos"])
        self.full = bool(state["full"])


def tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def masked_argmax(values: torch.Tensor, mask: np.ndarray) -> int:
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
    masked_values = values.masked_fill(~mask_t, -1e9)
    return int(torch.argmax(masked_values).item())


def choose_dqn_action(
    net: QNetwork,
    obs: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    valid = np.flatnonzero(mask)
    if random.random() < epsilon:
        return int(np.random.choice(valid))
    with torch.no_grad():
        q_values = net(tensor(obs[None, :], device))[0]
    return masked_argmax(q_values, mask)


def choose_a2c_action(
    net: ActorCritic,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, value = net(tensor(obs[None, :], device))
    mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
    masked_logits = logits.masked_fill(~mask_t, -1e9)
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    return int(action.item()), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0), value.squeeze(0)


def greedy_action_from_q(net: QNetwork, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        q_values = net(tensor(obs[None, :], device))[0]
    return masked_argmax(q_values, mask)


def greedy_action_from_actor(net: ActorCritic, obs: np.ndarray, mask: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        logits, _ = net(tensor(obs[None, :], device))
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
    return int(torch.argmax(masked_logits, dim=-1).item())


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0:
        return end
    progress = min(step / duration, 1.0)
    return start + progress * (end - start)


def summarize_episode(info: dict[str, Any]) -> dict[str, Any]:
    rows = info.get("episode_rows", [])
    labels = [str(row.get("response_label", "")) for row in rows]
    return {
        "student_id": info.get("student_id", ""),
        "initial_student_ability": info.get("initial_student_ability", ""),
        "initial_effective_ability": info.get("initial_effective_ability", ""),
        "final_effective_ability": info.get("effective_student_ability", ""),
        "total_agent_reward": info.get("total_agent_reward", ""),
        "total_student_reward": info.get("total_student_reward", ""),
        "accuracy": info.get("accuracy", ""),
        "total_time_taken": info.get("total_time_taken", ""),
        "invalid_action_count": sum(bool(row.get("invalid_action", False)) for row in rows),
        "question_sequence": " ".join(str(row.get("question_id", "")) for row in rows),
        "inherent_difficulty_sequence": " ".join(str(row.get("inherent_difficulty", "")) for row in rows),
        "target_difficulty_sequence": " ".join(str(row.get("target_inherent_difficulty", "")) for row in rows),
        "difficulty_gap_sequence": " ".join(str(row.get("difficulty_gap", "")) for row in rows),
        "difficulty_reward_sequence": " ".join(str(row.get("difficulty_match_reward", "")) for row in rows),
        "topic_reward_sequence": " ".join(str(row.get("topic_reward", "")) for row in rows),
        "diversity_reward_sequence": " ".join(str(row.get("diversity_reward", "")) for row in rows),
        "response_label_sequence": " ".join(labels),
        "acting_mode_sequence": " ".join(str(row.get("acting_mode", "")) for row in rows),
        "acting_ability_sequence": " ".join(str(row.get("acting_ability", "")) for row in rows),
        "suggested_learning_path": json.dumps(info.get("suggested_learning_path", []), ensure_ascii=False),
    }


@dataclass
class EvalSummary:
    eval_mean_agent_reward: float
    eval_mean_student_reward: float
    eval_mean_accuracy: float
    eval_mean_time_taken: float
    eval_mean_time_ratio: float
    eval_invalid_action_rate: float
    eval_effective_ability_start: float
    eval_effective_ability_end: float
    eval_effective_ability_delta: float
    eval_episode_length: float


def evaluate(
    algo: str,
    model: QNetwork | ActorCritic,
    eval_students: list[dict[str, Any]],
    episode_length: int,
    episodes_per_student: int,
    seed: int,
    device: torch.device,
) -> EvalSummary:
    agent_rewards = []
    student_rewards = []
    accuracies = []
    time_taken = []
    time_ratios = []
    invalid_rates = []
    ability_start = []
    ability_end = []
    lengths = []

    for student_idx, student in enumerate(eval_students):
        env = make_env([student], episode_length, seed + student_idx * 1000)

        for episode_idx in range(episodes_per_student):
            obs, _ = env.reset(seed=seed + student_idx * 1000 + episode_idx)
            done = False
            final_info: dict[str, Any] = {}

            while not done:
                mask = valid_action_mask(env)
                if algo == "dqn":
                    action = greedy_action_from_q(model, obs, mask, device)  # type: ignore[arg-type]
                else:
                    action = greedy_action_from_actor(model, obs, mask, device)  # type: ignore[arg-type]
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_info = info

            rows = final_info.get("episode_rows", [])
            agent_rewards.append(float(final_info.get("total_agent_reward", 0.0)))
            student_rewards.append(float(final_info.get("total_student_reward", 0.0)))
            accuracies.append(float(final_info.get("accuracy", 0.0)))
            time_taken.append(float(final_info.get("total_time_taken", 0.0)))
            time_ratios.append(float(np.mean([row.get("time_ratio", 0.0) for row in rows])) if rows else 0.0)
            invalid_rates.append(sum(bool(row.get("invalid_action", False)) for row in rows) / max(len(rows), 1))
            ability_start.append(float(final_info.get("initial_effective_ability", 0.0)))
            ability_end.append(float(final_info.get("effective_student_ability", 0.0)))
            lengths.append(len(rows))

        env.close()

    start = float(np.mean(ability_start))
    end = float(np.mean(ability_end))
    return EvalSummary(
        eval_mean_agent_reward=float(np.mean(agent_rewards)),
        eval_mean_student_reward=float(np.mean(student_rewards)),
        eval_mean_accuracy=float(np.mean(accuracies)),
        eval_mean_time_taken=float(np.mean(time_taken)),
        eval_mean_time_ratio=float(np.mean(time_ratios)),
        eval_invalid_action_rate=float(np.mean(invalid_rates)),
        eval_effective_ability_start=start,
        eval_effective_ability_end=end,
        eval_effective_ability_delta=end - start,
        eval_episode_length=float(np.mean(lengths)),
    )


def save_checkpoint(
    path: Path,
    algo: str,
    step: int,
    episode: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer | None,
    args: argparse.Namespace,
) -> None:
    payload = {
        "algo": algo,
        "step": step,
        "episode": episode,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
    }
    if replay is not None:
        payload["replay_buffer"] = replay.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer | None,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if replay is not None and "replay_buffer" in checkpoint:
        replay.load_state_dict(checkpoint["replay_buffer"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("episode", 0))


def train_dqn(args: argparse.Namespace, paths: dict[str, Path], run_id: str, device: torch.device) -> None:
    train_students = create_student_population(args.student_variants_per_ability, seed=args.student_profile_seed)
    eval_students = [create_student(f"eval_S{ability}", ability=ability) for ability in EVAL_ABILITIES]
    env = make_env(train_students, args.episode_length, args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions, args.hidden_dim).to(device)
    target_net = QNetwork(obs_dim, n_actions, args.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.replay_size, obs_dim)

    start_step = 0
    episode = 0
    if args.resume_from:
        start_step, episode = load_checkpoint(args.resume_from, q_net, optimizer, replay, device)
        target_net.load_state_dict(q_net.state_dict())

    obs, _ = env.reset(seed=args.seed + start_step)
    final_step = start_step + args.timesteps

    print("Custom dynamic DQN")
    print(f"  device: {device}")
    print(f"  obs_dim: {obs_dim}")
    print(f"  actions/questions: {n_actions}")
    print(f"  students: {len(train_students)}")

    for step in range(start_step + 1, final_step + 1):
        epsilon = linear_schedule(args.exploration_initial_eps, args.exploration_final_eps, args.exploration_fraction_steps, step)
        action = choose_dqn_action(q_net, obs, valid_action_mask(env), epsilon, device)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if len(replay) >= args.learning_starts and step % args.train_freq == 0:
            batch = replay.sample(args.batch_size)
            obs_t = tensor(batch["obs"], device)
            actions_t = tensor(batch["actions"], device, torch.long)
            rewards_t = tensor(batch["rewards"], device)
            next_obs_t = tensor(batch["next_obs"], device)
            dones_t = tensor(batch["dones"], device)

            q_values = q_net(obs_t).gather(1, actions_t[:, None]).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_obs_t)
                next_masks = np.stack(
                    [valid_action_mask_from_obs(row, n_actions) for row in batch["next_obs"]]
                )
                next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=device)
                next_q = next_q.masked_fill(~next_masks_t, -1e9)
                target = rewards_t + args.gamma * (1.0 - dones_t) * next_q.max(dim=1).values
            loss = F.smooth_l1_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
            optimizer.step()

        if step % args.target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            episode += 1
            row = {
                "run_id": run_id,
                "timestep": step,
                "episode": episode,
                **summarize_episode(info),
            }
            append_csv(paths["logs"] / "training_episode_summary.csv", row, TRAIN_EPISODE_FIELDS)
            obs, _ = env.reset()

        if step % args.eval_freq == 0:
            summary = evaluate("dqn", q_net, eval_students, args.episode_length, args.eval_episodes_per_student, args.seed + step, device)
            row = {"run_id": run_id, "timestep": step, "algo": "dqn", **asdict(summary)}
            append_csv(paths["logs"] / "evaluation_metrics.csv", row, EVAL_FIELDS)
            print(f"[eval] step={step} reward={summary.eval_mean_agent_reward:.3f} accuracy={summary.eval_mean_accuracy:.3f}")

        if step % args.checkpoint_freq == 0:
            save_checkpoint(paths["checkpoints"] / f"dqn_step_{step}.pt", "dqn", step, episode, q_net, optimizer, replay, args)
            print(f"[checkpoint] saved {paths['checkpoints'] / f'dqn_step_{step}.pt'}")

    save_checkpoint(paths["models"] / "final_model.pt", "dqn", final_step, episode, q_net, optimizer, replay, args)
    env.close()


def train_a2c(args: argparse.Namespace, paths: dict[str, Path], run_id: str, device: torch.device) -> None:
    train_students = create_student_population(args.student_variants_per_ability, seed=args.student_profile_seed)
    eval_students = [create_student(f"eval_S{ability}", ability=ability) for ability in EVAL_ABILITIES]
    env = make_env(train_students, args.episode_length, args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = ActorCritic(obs_dim, n_actions, args.hidden_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    start_step = 0
    episode = 0
    if args.resume_from:
        start_step, episode = load_checkpoint(args.resume_from, net, optimizer, None, device)

    obs, _ = env.reset(seed=args.seed + start_step)
    final_step = start_step + args.timesteps
    rollout: list[dict[str, Any]] = []

    print("Custom dynamic A2C")
    print(f"  device: {device}")
    print(f"  obs_dim: {obs_dim}")
    print(f"  actions/questions: {n_actions}")
    print(f"  students: {len(train_students)}")

    for step in range(start_step + 1, final_step + 1):
        action, log_prob, entropy, value = choose_a2c_action(net, obs, valid_action_mask(env), device)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rollout.append(
            {
                "reward": float(reward),
                "done": done,
                "log_prob": log_prob,
                "entropy": entropy,
                "value": value,
                "next_obs": next_obs,
            }
        )
        obs = next_obs

        should_update = len(rollout) >= args.rollout_steps or done
        if should_update:
            with torch.no_grad():
                if done:
                    next_value = torch.zeros((), device=device)
                else:
                    _, next_value_batch = net(tensor(obs[None, :], device))
                    next_value = next_value_batch.squeeze(0)

            returns = []
            running_return = next_value
            for item in reversed(rollout):
                running_return = torch.as_tensor(item["reward"], device=device) + args.gamma * running_return * (1.0 - float(item["done"]))
                returns.append(running_return)
            returns.reverse()

            returns_t = torch.stack(returns)
            values_t = torch.stack([item["value"] for item in rollout])
            log_probs_t = torch.stack([item["log_prob"] for item in rollout])
            entropy_t = torch.stack([item["entropy"] for item in rollout])
            advantages = returns_t - values_t

            actor_loss = -(log_probs_t * advantages.detach()).mean()
            critic_loss = F.mse_loss(values_t, returns_t.detach())
            entropy_bonus = entropy_t.mean()
            loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            rollout.clear()

        if done:
            episode += 1
            row = {
                "run_id": run_id,
                "timestep": step,
                "episode": episode,
                **summarize_episode(info),
            }
            append_csv(paths["logs"] / "training_episode_summary.csv", row, TRAIN_EPISODE_FIELDS)
            obs, _ = env.reset()

        if step % args.eval_freq == 0:
            summary = evaluate("a2c", net, eval_students, args.episode_length, args.eval_episodes_per_student, args.seed + step, device)
            row = {"run_id": run_id, "timestep": step, "algo": "a2c", **asdict(summary)}
            append_csv(paths["logs"] / "evaluation_metrics.csv", row, EVAL_FIELDS)
            print(f"[eval] step={step} reward={summary.eval_mean_agent_reward:.3f} accuracy={summary.eval_mean_accuracy:.3f}")

        if step % args.checkpoint_freq == 0:
            save_checkpoint(paths["checkpoints"] / f"a2c_step_{step}.pt", "a2c", step, episode, net, optimizer, None, args)
            print(f"[checkpoint] saved {paths['checkpoints'] / f'a2c_step_{step}.pt'}")

    save_checkpoint(paths["models"] / "final_model.pt", "a2c", final_step, episode, net, optimizer, None, args)
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom DQN/A2C on the dynamic adaptive MCQ env.")
    parser.add_argument("--algo", choices=["dqn", "a2c"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/dynamic_custom_dqn"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episode-length", type=int, default=15)
    parser.add_argument("--student-variants-per-ability", type=int, default=5)
    parser.add_argument("--student-profile-seed", type=int, default=123)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes-per-student", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--resume-from", type=Path, default=None)

    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-freq", type=int, default=1_000)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--exploration-fraction-steps", type=int, default=250_000)

    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    run_id = make_run_id()
    paths = make_dirs(args.save_dir)

    config = {
        "run_id": run_id,
        "algo": args.algo,
        "timesteps": args.timesteps,
        "ability_range": f"{MIN_ABILITY}-{MAX_ABILITY}",
        "simulator": "dynamic_student_simulator.py",
        "env": "dynamic_ability_env.py",
        "uses_stable_baselines3": False,
        "args": vars(args),
    }
    (paths["root"] / "training_config.json").write_text(
        json.dumps(config, indent=2, default=json_default),
        encoding="utf-8",
    )

    if args.algo == "dqn":
        train_dqn(args, paths, run_id, device)
    else:
        train_a2c(args, paths, run_id, device)

    print(f"Final model path: {paths['models'] / 'final_model.pt'}")


if __name__ == "__main__":
    main()

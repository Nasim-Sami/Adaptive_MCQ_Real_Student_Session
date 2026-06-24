"""
Recurrent Double DQN (DRQN-style, LSTM) trainer for the topic-selection env -
the from-scratch counterpart to ``train_double_dqn.py``, extended with memory
the same way ``train_recurrent_ppo.py`` / ``train_a2c_lstm.py`` are.

Why recurrent: see train_a2c_lstm.py / train_recurrent_ppo.py module docstrings
- the hidden learning style and forgetting rate can only be inferred from the
history of answers, which a memoryless Q-network cannot do.

Replay buffer design: classic DRQN needs padding/burn-in machinery because
episodes vary in length. Here every episode is FIXED LENGTH
(sub_episode_length * n_sub_episodes - see TopicSelectionMCQEnv.step, which
always terminates exactly there), so ``rl_common.EpisodeReplayBuffer`` stores
whole episodes as fixed-size rows with no padding at all. Training samples a
batch of complete episodes and unrolls the LSTM once per sequence:
  - online net is unrolled over ``obs`` (the stored sequence) for Q(s, a)
  - online net is unrolled over ``next_obs`` (a second, independent zero-state
    unroll) to pick the Double-DQN argmax action
  - target net is unrolled over ``next_obs`` the same way to evaluate it
This is the standard simplification used when a stored recurrent state isn't
tracked (R2D2-style stored-state is more accurate but unnecessary at this
episode length / problem size).

Run (quick smoke test):
    python train_double_dqn_lstm.py --timesteps 4000 --sub-episode-length 20 \\
        --n-sub-episodes 4 --eval-freq 2000 --learning-starts-episodes 5 \\
        --save-dir runs/double_dqn_lstm_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rl_common as rc
import student_simulator as sim
from mcq_env import make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recurrent (DRQN) Double DQN (topic selection).")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--save-dir", type=Path, default=Path("runs/double_dqn_lstm"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--sub-episode-length", type=int, default=20)
    p.add_argument("--n-sub-episodes", type=int, default=4)
    p.add_argument("--student-variants-per-ability", type=int, default=4)
    p.add_argument("--student-profile-seed", type=int, default=123)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    # replay / DQN - sized in EPISODES, not steps (each row is a full episode)
    p.add_argument("--replay-size-episodes", type=int, default=2_000)
    p.add_argument("--batch-size-episodes", type=int, default=32)
    p.add_argument("--learning-starts-episodes", type=int, default=50)
    p.add_argument("--train-freq-episodes", type=int, default=1)
    p.add_argument("--target-update-freq-episodes", type=int, default=20)
    p.add_argument("--exploration-initial-eps", type=float, default=1.0)
    p.add_argument("--exploration-final-eps", type=float, default=0.05)
    p.add_argument("--exploration-fraction-steps", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=20_000)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Path to a training checkpoint (checkpoints/double_dqn_lstm_ckpt.pt) to resume from.")
    return p.parse_args()


def epsilon_greedy_topic_recurrent(net, obs, mask, hidden, epsilon, device):
    if np.random.random() < epsilon:
        valid = np.flatnonzero(mask)
        action = int(np.random.choice(valid))
        with torch.no_grad():
            _, new_hidden = net(rc.tensor(obs[None, None, :], device), hidden)
        return action, new_hidden
    return rc.greedy_topic_from_recurrent_q(net, obs, mask, hidden, device)


def main() -> None:
    args = parse_args()
    rc.seed_everything(args.seed)
    device = rc.get_device(args.device)
    run_id = rc.make_run_id()
    paths = rc.make_dirs(args.save_dir)

    students = sim.create_student_population(args.student_variants_per_ability, seed=args.student_profile_seed)
    env = make_env(students=students, sub_episode_length=args.sub_episode_length,
                   n_sub_episodes=args.n_sub_episodes, seed=args.seed, randomize_initial_ability=True)
    obs_dim, n_actions = int(env.obs_dim), int(env.action_space.n)
    episode_len = env.episode_length

    q_net = rc.RecurrentQNetwork(obs_dim, n_actions, args.hidden_dim, args.lstm_hidden).to(device)
    target_net = rc.RecurrentQNetwork(obs_dim, n_actions, args.hidden_dim, args.lstm_hidden).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay = rc.EpisodeReplayBuffer(args.replay_size_episodes, episode_len, obs_dim, n_actions)

    (paths["root"] / "training_config.json").write_text(
        json.dumps({"algo": "double_dqn_lstm", "run_id": run_id, "uses_stable_baselines3": False,
                    "obs_dim": obs_dim, "n_actions": n_actions, "episode_len": episode_len,
                    "args": {k: str(v) for k, v in vars(args).items()}}, indent=2), encoding="utf-8")

    start_step, episode, best_eval = 0, 0, -float("inf")
    if args.resume_from:
        start_step, episode, best_eval = rc.load_train_checkpoint(
            args.resume_from, model=q_net, optimizer=optimizer, device=device, replay=replay)
        target_net.load_state_dict(q_net.state_dict())
        print(f"Resumed from {args.resume_from} at step={start_step} episode={episode} best_eval={best_eval:.3f}")

    print(f"Double DQN+LSTM | device={device} obs_dim={obs_dim} topics(actions)={n_actions} "
          f"episode={args.n_sub_episodes}x{args.sub_episode_length} (len={episode_len})")

    obs, reset_info = env.reset(seed=args.seed + start_step)
    hidden = q_net.init_hidden(1, device)
    ep_obs = np.zeros((episode_len, obs_dim), dtype=np.float32)
    ep_next_obs = np.zeros((episode_len, obs_dim), dtype=np.float32)
    ep_actions = np.zeros(episode_len, dtype=np.int64)
    ep_rewards = np.zeros(episode_len, dtype=np.float32)
    ep_dones = np.zeros(episode_len, dtype=np.float32)
    ep_next_masks = np.zeros((episode_len, n_actions), dtype=np.float32)
    t_in_episode = 0

    final_step = start_step + args.timesteps
    for step in range(start_step + 1, final_step + 1):
        epsilon = rc.linear_schedule(args.exploration_initial_eps, args.exploration_final_eps,
                                     args.exploration_fraction_steps, step)
        mask = rc.valid_topic_mask(env)
        action, hidden = epsilon_greedy_topic_recurrent(q_net, obs, mask, hidden, epsilon, device)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = rc.valid_topic_mask(env).astype(np.float32)

        ep_obs[t_in_episode] = obs
        ep_next_obs[t_in_episode] = next_obs
        ep_actions[t_in_episode] = action
        ep_rewards[t_in_episode] = reward
        ep_dones[t_in_episode] = float(done)
        ep_next_masks[t_in_episode] = next_mask
        t_in_episode += 1
        obs = next_obs

        if done:
            replay.add_episode(ep_obs, ep_actions, ep_rewards, ep_next_obs, ep_dones, ep_next_masks)
            episode += 1

            # ---- Double DQN update over a batch of FULL stored episodes ----
            if len(replay) >= args.learning_starts_episodes and episode % args.train_freq_episodes == 0:
                batch = replay.sample(args.batch_size_episodes)
                obs_t = rc.tensor(batch["obs"], device)              # (B, T, obs_dim)
                next_obs_t = rc.tensor(batch["next_obs"], device)    # (B, T, obs_dim)
                actions_t = rc.tensor(batch["actions"], device, torch.long)
                rewards_t = rc.tensor(batch["rewards"], device)
                dones_t = rc.tensor(batch["dones"], device)
                next_masks_t = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=device)

                q_all, _ = q_net(obs_t)                              # (B, T, n_actions)
                q_sa = q_all.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
                with torch.no_grad():
                    online_next_q, _ = q_net(next_obs_t)
                    online_next_q = online_next_q.masked_fill(~next_masks_t, -1e9)
                    next_actions = online_next_q.argmax(dim=2, keepdim=True)
                    target_next_q_all, _ = target_net(next_obs_t)
                    target_next_q = target_next_q_all.gather(2, next_actions).squeeze(-1)
                    target = rewards_t + args.gamma * (1.0 - dones_t) * target_next_q
                loss = F.smooth_l1_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
                optimizer.step()

            if episode % args.target_update_freq_episodes == 0:
                target_net.load_state_dict(q_net.state_dict())

            row = {"run_id": run_id, "timestep": step, "episode": episode, "algo": "double_dqn_lstm",
                   **rc.summarize_training_episode(info, reset_info)}
            rc.append_csv(paths["logs"] / "training_episodes.csv", row, rc.TRAIN_EPISODE_FIELDS)
            obs, reset_info = env.reset()
            hidden = q_net.init_hidden(1, device)  # fresh student -> fresh memory
            t_in_episode = 0

        if step % args.eval_freq == 0:
            metrics = rc.evaluate_recurrent(q_net, "double_dqn_lstm", n_episodes=args.eval_episodes,
                                           sub_episode_length=args.sub_episode_length,
                                           n_sub_episodes=args.n_sub_episodes, seed=args.seed + step,
                                           device=device, students=students)
            rc.append_csv(paths["logs"] / "evaluation.csv",
                          {"run_id": run_id, "timestep": step, "algo": "double_dqn_lstm", **metrics}, rc.EVAL_FIELDS)
            print(f"[eval] step={step} reward={metrics['eval_mean_agent_reward']:.3f} "
                  f"mastery_gain={metrics['eval_mean_mastery_improvement']:+.4f} "
                  f"acc={metrics['eval_mean_accuracy']:.3f} repeat_rate={metrics['eval_repeat_rate']:.3f}")
            if metrics["eval_mean_mastery_improvement"] > best_eval:
                best_eval = metrics["eval_mean_mastery_improvement"]
                rc.save_model(paths["models"] / "best_model.pt", algo="double_dqn_lstm", model=q_net,
                              obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden_dim,
                              lstm_hidden=args.lstm_hidden, extra={"step": step, "eval_mastery_gain": best_eval})
                print(f"        new best true_mastery_gain={best_eval:+.4f} -> saved best_model.pt")

        if step % args.checkpoint_freq == 0:
            rc.save_train_checkpoint(paths["checkpoints"] / "double_dqn_lstm_ckpt.pt", algo="double_dqn_lstm",
                                     step=step, episode=episode, best_eval=best_eval, model=q_net,
                                     optimizer=optimizer, obs_dim=obs_dim, n_actions=n_actions,
                                     hidden=args.hidden_dim, lstm_hidden=args.lstm_hidden, replay=replay)
            print(f"[checkpoint] training state saved at step {step} -> checkpoints/double_dqn_lstm_ckpt.pt")

    rc.save_train_checkpoint(paths["checkpoints"] / "double_dqn_lstm_ckpt.pt", algo="double_dqn_lstm",
                             step=final_step, episode=episode, best_eval=best_eval, model=q_net,
                             optimizer=optimizer, obs_dim=obs_dim, n_actions=n_actions,
                             hidden=args.hidden_dim, lstm_hidden=args.lstm_hidden, replay=replay)
    rc.save_model(paths["models"] / "final_model.pt", algo="double_dqn_lstm", model=q_net,
                  obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden_dim,
                  lstm_hidden=args.lstm_hidden, extra={"step": final_step})
    print(f"Saved final model -> {paths['models'] / 'final_model.pt'} "
          f"(resume with --resume-from {paths['checkpoints'] / 'double_dqn_lstm_ckpt.pt'})")

    # ---- final scoreboard against heuristics ---------------------------
    print("\n================  FINAL SCOREBOARD  ================")
    import evaluate_baselines as eb
    policies = eb.default_heuristics()
    policies["trained[double_dqn_lstm]"] = rc.RecurrentTorchPolicyWrapper(q_net, "double_dqn_lstm", device)
    eb.print_scoreboard(
        policies, episodes=200, seed=98_765,
        sub_episode_length=args.sub_episode_length, n_sub_episodes=args.n_sub_episodes,
        mastery_threshold=0.6, students=students,
    )
    print("\nGoal:  random << best_heuristic << trained  on TRUE_mastery_gain.\n")


if __name__ == "__main__":
    main()

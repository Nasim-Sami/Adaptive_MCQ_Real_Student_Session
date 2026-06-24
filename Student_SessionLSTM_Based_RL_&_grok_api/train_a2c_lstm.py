"""
Recurrent A2C (LSTM) trainer for the topic-selection env - the from-scratch
counterpart to ``train_recurrent_ppo.py`` (sb3-contrib), kept in the same
hand-rolled style as ``train_a2c.py`` (no stable-baselines).

Why recurrent: the student's learning style and forgetting rate are HIDDEN.
A memoryless policy cannot tell a 'massed' learner from an 'interleaved' one
on the first question - they look identical. An LSTM integrates the history
of (topic, correct?, time, recency) so the policy can *infer* the latent
style and adapt sequencing, which a fixed rule cannot do.

Update rule: because every episode is FIXED LENGTH (sub_episode_length *
n_sub_episodes, see TopicSelectionMCQEnv.step), one episode = one independent
LSTM sequence. We run the LSTM hidden state from zero at episode start,
collect the whole episode, then do ONE full-sequence A2C update (true BPTT
through the episode) and reset the hidden state to zero for the next student.
This is simpler and just as correct as a windowed-rollout scheme here, since
there is no "continuing" episode to carry state across.

Run (quick smoke test):
    python train_a2c_lstm.py --timesteps 4000 --sub-episode-length 20 \\
        --n-sub-episodes 4 --eval-freq 2000 --save-dir runs/a2c_lstm_smoke
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
from torch.distributions import Categorical

import rl_common as rc
import student_simulator as sim
from mcq_env import make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recurrent (LSTM) A2C (topic selection).")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--save-dir", type=Path, default=Path("runs/a2c_lstm"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--sub-episode-length", type=int, default=20)
    p.add_argument("--n-sub-episodes", type=int, default=4)
    p.add_argument("--student-variants-per-ability", type=int, default=4)
    p.add_argument("--student-profile-seed", type=int, default=123)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=20_000)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Path to a training checkpoint (checkpoints/a2c_lstm_ckpt.pt) to resume from.")
    return p.parse_args()


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

    net = rc.RecurrentActorCritic(obs_dim, n_actions, args.hidden_dim, args.lstm_hidden).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    (paths["root"] / "training_config.json").write_text(
        json.dumps({"algo": "a2c_lstm", "run_id": run_id, "uses_stable_baselines3": False,
                    "obs_dim": obs_dim, "n_actions": n_actions, "args": {k: str(v) for k, v in vars(args).items()}},
                   indent=2), encoding="utf-8")

    start_step, episode, best_eval = 0, 0, -float("inf")
    if args.resume_from:
        start_step, episode, best_eval = rc.load_train_checkpoint(
            args.resume_from, model=net, optimizer=optimizer, device=device, replay=None)
        print(f"Resumed from {args.resume_from} at step={start_step} episode={episode} best_eval={best_eval:.3f}")

    print(f"A2C+LSTM | device={device} obs_dim={obs_dim} topics(actions)={n_actions} "
          f"episode={args.n_sub_episodes}x{args.sub_episode_length}")

    obs, reset_info = env.reset(seed=args.seed + start_step)
    hidden = net.init_hidden(1, device)
    episode_buf: list[dict] = []
    final_step = start_step + args.timesteps
    for step in range(start_step + 1, final_step + 1):
        mask = rc.valid_topic_mask(env)
        obs_t = rc.tensor(obs[None, None, :], device)
        logits, value, hidden = net(obs_t, hidden)
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        dist = Categorical(logits=logits[0, 0].masked_fill(~mask_t[0], -1e9))
        action = dist.sample()

        next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
        done = terminated or truncated
        episode_buf.append({
            "reward": float(reward), "done": done,
            "log_prob": dist.log_prob(action), "entropy": dist.entropy(),
            "value": value[0, 0],
        })
        obs = next_obs

        if done:
            # full episode collected -> one BPTT update over the WHOLE sequence
            with torch.no_grad():
                next_value = torch.zeros((), device=device)  # always a true terminal (fixed-length episode)
            returns = []
            running = next_value
            for item in reversed(episode_buf):
                running = torch.as_tensor(item["reward"], device=device) + args.gamma * running * (1.0 - float(item["done"]))
                returns.append(running)
            returns.reverse()

            returns_t = torch.stack(returns)
            values_t = torch.stack([it["value"] for it in episode_buf])
            log_probs_t = torch.stack([it["log_prob"] for it in episode_buf])
            entropy_t = torch.stack([it["entropy"] for it in episode_buf])
            advantages = returns_t - values_t

            actor_loss = -(log_probs_t * advantages.detach()).mean()
            critic_loss = F.mse_loss(values_t, returns_t.detach())
            loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy_t.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            episode_buf.clear()

            episode += 1
            row = {"run_id": run_id, "timestep": step, "episode": episode, "algo": "a2c_lstm",
                   **rc.summarize_training_episode(info, reset_info)}
            rc.append_csv(paths["logs"] / "training_episodes.csv", row, rc.TRAIN_EPISODE_FIELDS)
            obs, reset_info = env.reset()
            hidden = net.init_hidden(1, device)  # fresh student -> fresh memory

        if step % args.eval_freq == 0:
            metrics = rc.evaluate_recurrent(net, "a2c_lstm", n_episodes=args.eval_episodes,
                                           sub_episode_length=args.sub_episode_length,
                                           n_sub_episodes=args.n_sub_episodes, seed=args.seed + step,
                                           device=device, students=students)
            rc.append_csv(paths["logs"] / "evaluation.csv",
                          {"run_id": run_id, "timestep": step, "algo": "a2c_lstm", **metrics}, rc.EVAL_FIELDS)
            print(f"[eval] step={step} reward={metrics['eval_mean_agent_reward']:.3f} "
                  f"mastery_gain={metrics['eval_mean_mastery_improvement']:+.4f} "
                  f"acc={metrics['eval_mean_accuracy']:.3f} repeat_rate={metrics['eval_repeat_rate']:.3f}")
            if metrics["eval_mean_mastery_improvement"] > best_eval:
                best_eval = metrics["eval_mean_mastery_improvement"]
                rc.save_model(paths["models"] / "best_model.pt", algo="a2c_lstm", model=net,
                              obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden_dim,
                              lstm_hidden=args.lstm_hidden, extra={"step": step, "eval_mastery_gain": best_eval})
                print(f"        new best true_mastery_gain={best_eval:+.4f} -> saved best_model.pt")

        if step % args.checkpoint_freq == 0:
            rc.save_train_checkpoint(paths["checkpoints"] / "a2c_lstm_ckpt.pt", algo="a2c_lstm", step=step,
                                     episode=episode, best_eval=best_eval, model=net, optimizer=optimizer,
                                     obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden_dim,
                                     lstm_hidden=args.lstm_hidden, replay=None)
            print(f"[checkpoint] training state saved at step {step} -> checkpoints/a2c_lstm_ckpt.pt")

    rc.save_train_checkpoint(paths["checkpoints"] / "a2c_lstm_ckpt.pt", algo="a2c_lstm", step=final_step,
                             episode=episode, best_eval=best_eval, model=net, optimizer=optimizer,
                             obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden_dim,
                             lstm_hidden=args.lstm_hidden, replay=None)
    rc.save_model(paths["models"] / "final_model.pt", algo="a2c_lstm", model=net,
                  obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden_dim,
                  lstm_hidden=args.lstm_hidden, extra={"step": final_step})
    print(f"Saved final model -> {paths['models'] / 'final_model.pt'} "
          f"(resume with --resume-from {paths['checkpoints'] / 'a2c_lstm_ckpt.pt'})")

    # ---- final scoreboard against heuristics (and RecurrentPPO if present) --
    print("\n================  FINAL SCOREBOARD  ================")
    import evaluate_baselines as eb
    policies = eb.default_heuristics()
    policies["trained[a2c_lstm]"] = rc.RecurrentTorchPolicyWrapper(net, "a2c_lstm", device)
    eb.print_scoreboard(
        policies, episodes=200, seed=98_765,
        sub_episode_length=args.sub_episode_length, n_sub_episodes=args.n_sub_episodes,
        mastery_threshold=0.6, students=students,
    )
    print("\nGoal:  random << best_heuristic << trained  on TRUE_mastery_gain.\n")


if __name__ == "__main__":
    main()

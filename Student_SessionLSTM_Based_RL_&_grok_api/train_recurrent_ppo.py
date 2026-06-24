"""
Recurrent PPO trainer for the topic-selection problem (the memory-equipped agent).

Why recurrent?  The student's learning style and forgetting rate are HIDDEN.  A
memoryless policy cannot tell a 'massed' learner from an 'interleaved' one - they
look identical on the first question.  An LSTM policy integrates the history of
(topic, correct?, time, recency) and can *infer* the latent style, then adapt its
sequencing.  That inference is exactly what a fixed rule cannot do, and it is the
headroom that lets RL beat the heuristics.

Masking note: with 15 questions per topic and an 80-step episode a topic is almost
never exhausted, so the action mask is nearly a no-op; the env already redirects
the rare invalid pick and penalises it.  We therefore use plain RecurrentPPO
(LSTM) rather than fighting to combine masking with recurrence.

Usage:
    python train_recurrent_ppo.py --timesteps 400000
    python train_recurrent_ppo.py --timesteps 1000000 --n-envs 8 --seed 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import student_simulator as sim
from mcq_env import make_env
import evaluate_baselines as eb


# --------------------------------------------------------------------------- #
# env factory
# --------------------------------------------------------------------------- #
def make_training_env(students, sub_len, n_sub, seed):
    def _thunk():
        return make_env(students=students, sub_episode_length=sub_len,
                        n_sub_episodes=n_sub, seed=seed, randomize_initial_ability=True)
    return _thunk


# --------------------------------------------------------------------------- #
# a stateful policy wrapper so the trained LSTM can be scored by the shared
# scoreboard harness (evaluate_baselines.run_policy / print_scoreboard)
# --------------------------------------------------------------------------- #
class RecurrentPolicyWrapper:
    """Adapts a RecurrentPPO model to the (env, obs, mask, rng) -> action API,
    carrying the LSTM hidden state across a single episode.

    deterministic=False is intentional.  PPO trains a stochastic policy, and with
    a non-trivial entropy coefficient the deterministic (argmax) action can
    collapse to a degenerate choice that looks far worse than the actual policy
    behaviour.  Sampling matches what the agent does at training/inference time
    and gives a faithful measure of policy quality.
    """

    def __init__(self, model: RecurrentPPO, *, deterministic: bool = False) -> None:
        self.model = model
        self.deterministic = bool(deterministic)
        self.state = None
        self._start = True

    def reset(self) -> None:
        self.state = None
        self._start = True

    def __call__(self, env, obs, mask, rng) -> int:
        action, self.state = self.model.predict(
            np.asarray(obs, dtype=np.float32),
            state=self.state,
            episode_start=np.array([self._start]),
            deterministic=self.deterministic,
        )
        self._start = False
        return int(action)


# --------------------------------------------------------------------------- #
# periodic evaluation on TRUE mastery gain; saves the best model
# --------------------------------------------------------------------------- #
class MasteryEvalCallback(BaseCallback):
    def __init__(self, *, students, sub_len, n_sub, eval_episodes, eval_freq,
                 seed, save_path: Path, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.students = students
        self.sub_len = sub_len
        self.n_sub = n_sub
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.seed = seed
        self.save_path = save_path
        self.best_gain = -1e9

    def _run_eval(self) -> dict:
        pol = RecurrentPolicyWrapper(self.model)
        return eb.run_policy(pol, episodes=self.eval_episodes, seed=self.seed,
                             sub_episode_length=self.sub_len, n_sub_episodes=self.n_sub,
                             mastery_threshold=0.6, students=self.students)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        res = self._run_eval()
        gain = float(res["mastery_gain"].mean())
        reward = float(res["reward"].mean())
        self.logger.record("eval/true_mastery_gain", gain)
        self.logger.record("eval/env_reward", reward)
        self.logger.record("eval/topics_mastered", float(res["topics_mastered"].mean()))
        if self.verbose:
            print(f"[eval @ {self.num_timesteps:>8d}]  true_mastery_gain={gain:+.4f}  "
                  f"env_reward={reward:+.2f}  #mastered={res['topics_mastered'].mean():.2f}")
        if gain > self.best_gain:
            self.best_gain = gain
            self.model.save(self.save_path / "best_model")
            if self.verbose:
                print(f"            new best true_mastery_gain={gain:+.4f}  (saved)")
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Recurrent PPO trainer (topic selection)")
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-episode-length", type=int, default=20)
    ap.add_argument("--n-sub-episodes", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--final-eval-episodes", type=int, default=200)
    ap.add_argument("--save-dir", type=str, default="runs/recurrent_ppo")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="path to a saved .zip checkpoint (e.g. runs/recurrent_ppo/final_model.zip) "
                         "to continue training from, instead of initializing fresh weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # one shared student population for train + eval (apples to apples with the
    # heuristic scoreboard, which also uses create_student_population)
    students = sim.create_student_population(variants_per_ability=4, seed=args.seed)

    venv = DummyVecEnv([
        make_training_env(students, args.sub_episode_length, args.n_sub_episodes, args.seed + i)
        for i in range(args.n_envs)
    ])
    venv = VecMonitor(venv)
    # Reward normalisation is NOT used here: episode totals (~5-18) are already in
    # a healthy range and the per-step reward is highly skewed (most steps near 0
    # plus occasional spikes when a topic crosses a learning threshold).  The
    # VecNormalize running-mean rescaling actively muddies that signal early in
    # training - removing it lets PPO see the raw learning gradient.

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = RecurrentPPO.load(args.resume_from, env=venv, device=args.device)
        # honor any hyperparameter overrides passed on this resumed run
        model.learning_rate = args.lr
        model.ent_coef = args.ent_coef
        model.gamma = args.gamma
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            venv,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            gae_lambda=0.95,
            n_epochs=10,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(save_path / "tb"),
            policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=1),
        )

    cb = MasteryEvalCallback(
        students=students, sub_len=args.sub_episode_length, n_sub=args.n_sub_episodes,
        eval_episodes=args.eval_episodes, eval_freq=max(1, args.eval_freq // args.n_envs),
        seed=10_000 + args.seed, save_path=save_path,
    )

    print(f"Training RecurrentPPO for {args.timesteps} steps "
          f"({args.n_envs} envs, obs_dim={venv.observation_space.shape[0]})...")
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=True,
                reset_num_timesteps=(args.resume_from is None))

    model.save(save_path / "final_model")

    # ---- final scoreboard: heuristics vs the trained recurrent agent --------
    print("\n================  FINAL SCOREBOARD  ================")
    policies = eb.default_heuristics()
    # use the BEST saved model if present, else the final one
    best_file = save_path / "best_model.zip"
    score_model = RecurrentPPO.load(best_file) if best_file.exists() else model
    policies["trained[recurrent_ppo]"] = RecurrentPolicyWrapper(score_model)
    eb.print_scoreboard(
        policies, episodes=args.final_eval_episodes, seed=98_765,
        sub_episode_length=args.sub_episode_length, n_sub_episodes=args.n_sub_episodes,
        mastery_threshold=0.6, students=students,
    )
    print("\nGoal:  random << best_heuristic << trained  on TRUE_mastery_gain.\n")


if __name__ == "__main__":
    main()

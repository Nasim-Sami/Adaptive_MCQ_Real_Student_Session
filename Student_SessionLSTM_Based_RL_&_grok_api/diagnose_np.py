"""Torch-free diagnosis: load an ActorCritic/QNetwork .pt and drive the real env.

Lets you inspect topic selection + difficulty adaptation WITHOUT installing torch
or setting up the LLM (bank questions only). Run from this folder:

    python diagnose_np.py [path/to/best_model.pt]      # default: A2C best_model
"""
import os, sys, zipfile, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); os.chdir(HERE)

DTYPE = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
         "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16}

class _S:
    def __init__(s, k, d, n): s.key, s.dtype, s.numel = k, d, n

def load_state_dict(path):
    zf = zipfile.ZipFile(path); prefix = zf.namelist()[0].split("/")[0]
    def rebuild(storage, off, size, stride, rg, hooks, *a):
        arr = np.frombuffer(zf.read(f"{prefix}/data/{storage.key}"), dtype=storage.dtype)
        n = int(np.prod(size)) if size else 1
        arr = arr[off:off + n]
        return arr.reshape(size) if size else arr
    class U(pickle.Unpickler):
        def find_class(self, m, n):
            if n.endswith("Storage"): return ("ST", n)
            if m == "torch._utils" and n == "_rebuild_tensor_v2": return rebuild
            if m == "collections" and n == "OrderedDict":
                from collections import OrderedDict; return OrderedDict
            try: return super().find_class(m, n)
            except Exception: return dict
        def persistent_load(self, pid):
            typ, key, numel = pid[1], pid[2], pid[4]
            dt = DTYPE.get(typ[1], np.float32) if isinstance(typ, tuple) else np.float32
            return _S(str(key), dt, numel)
    return U(zf.open(f"{prefix}/data.pkl")).load()

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs/a2c/models/best_model.pt")
payload = load_state_dict(MODEL); sd = payload["model_state"]; algo = payload.get("algo", "a2c")
if algo == "a2c":
    W0, b0 = np.asarray(sd["body.0.weight"]), np.asarray(sd["body.0.bias"])
    W2, b2 = np.asarray(sd["body.2.weight"]), np.asarray(sd["body.2.bias"])
    Wa, ba = np.asarray(sd["actor.weight"]), np.asarray(sd["actor.bias"])
else:  # dqn: net.0/2/4
    W0, b0 = np.asarray(sd["net.0.weight"]), np.asarray(sd["net.0.bias"])
    W2, b2 = np.asarray(sd["net.2.weight"]), np.asarray(sd["net.2.bias"])
    Wa, ba = np.asarray(sd["net.4.weight"]), np.asarray(sd["net.4.bias"])
print(f"loaded {algo.upper()}: obs_dim={W0.shape[1]} hidden={W0.shape[0]} n_actions={Wa.shape[0]}")

def greedy_topic(obs, mask):
    h = np.maximum(0.0, W0 @ obs + b0); h = np.maximum(0.0, W2 @ h + b2)
    z = (Wa @ h + ba).copy(); z[~mask] = -1e9
    return int(np.argmax(z))

import student_simulator as sim
from mcq_env import make_env

def run(init_ability, answer_fn, seed=7, carryover=None, label=""):
    env = make_env(sub_episode_length=20, n_sub_episodes=1, seed=seed, randomize_initial_ability=False)
    env.reset(seed=seed)
    env.effective_ability = float(sim.clip(init_ability, sim.MIN_ABILITY, sim.MAX_ABILITY))
    env.initial_effective_ability = env.effective_ability
    import difficulty_control as dc
    env.target_difficulty = dc.initial_target_difficulty(env.effective_ability)
    if carryover:
        for t, st in carryover.items():
            ti = env.topic_to_idx.get(t)
            if ti is not None:
                env.topic_asked[ti] = st["asked"]; env.topic_correct[ti] = st["correct"]; env.topic_wrong[ti] = st["wrong"]
    rng = np.random.default_rng(seed); rows = []
    for i in range(20):
        obs = env._get_obs().astype(np.float64); mask = env.valid_topic_mask().astype(bool)
        ti = greedy_topic(obs, mask); q_global, was_repeat, target = env.select_question_in_topic(ti)
        q = env.questions[q_global]; ok, t = answer_fn(q, target, env.effective_ability, rng)
        info = env.apply_external_answer(ti, q_global, ok, t)
        rows.append(dict(topic=env.topics[ti], diff=round(float(q["inherent_difficulty"]), 1),
                         tgt=round(float(target), 1), ok=ok,
                         a0=info["effective_ability_before"], a1=info["effective_ability_after"]))
    print("\n" + "=" * 78 + f"\n{label}\n" + "=" * 78)
    for j, r in enumerate(rows, 1):
        print(f"{j:>2} {r['topic'][:32]:<33} diff {r['diff']:>4} tgt {r['tgt']:>4} "
              f"{'Y' if r['ok'] else 'n'} {r['a0']:>6.2f}->{r['a1']:>5.2f}")
    uniq = len(dict.fromkeys(r["topic"] for r in rows))
    print(f"  unique topics {uniq}/20 | acc {sum(r['ok'] for r in rows)}/20 | "
          f"ability {rows[0]['a0']:.1f}->{rows[-1]['a1']:.1f}")

def sim_ans(ability, style=None, seed=0):
    rng = np.random.default_rng(seed)
    s = sim.create_student("sim", ability=int(round(ability)), rng=rng, learning_style=style)
    def fn(q, target, eff, _):
        st = sim.with_ability(s, eff); st["topic_mastery"] = s["topic_mastery"]
        r = sim.simulate_answer(st, q, rng)
        s.update(sim.apply_learning_update(s, q, bool(r["is_correct"]),
                 r["time_taken"]/max(q["base_time"], 1e-6), rng)["student"])
        return bool(r["is_correct"]), float(r["time_taken"])
    return fn

def human_ans(q, target, eff, _):  # a strong human: correct except hardest
    d = float(q["inherent_difficulty"])
    ok = True if d <= 8.5 else (hash(q["question_id"]) % 2 == 0)
    return ok, 8 + d * 7.5

if __name__ == "__main__":
    run(20, human_ans, label="HUMAN (answers correctly, start ability 20)")
    run(30, sim_ans(30, "interleaved", 1), label="SIMULATED average student (ability 30)")
    run(22, sim_ans(22, "interleaved", 1), label="SIMULATED weaker student (ability 22)")

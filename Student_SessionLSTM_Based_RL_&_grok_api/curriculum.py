"""
Curriculum graph for the June-15 adaptive-mechatronics RL system.

This module defines the *structure* of the domain that the RL agent reasons over:

  * TOPICS            - the 45 canonical mechatronics topics (no sub-topics).
  * TOPIC_CLUSTERS    - chapter-style groupings (used by the "blocked" learner).
  * PREREQUISITES     - a Directed Acyclic Graph (DAG): which topics should be
                        understood before a given topic. Sequencing now matters.
  * TRANSFER_MATRIX   - directed (A -> B) mastery spill-over. Positive values are
                        transfer (studying A helps B); negative values are
                        interference (studying A slightly hurts B).

The topics are listed in a *topological order*: every prerequisite of a topic
appears earlier in the TOPICS list than the topic itself. That guarantees the
PREREQUISITES graph is acyclic (validated by ``validate_dag`` / the __main__
self-test at the bottom of this file).

Curriculum scope follows R.K. Rajput, "A Textbook of Mechatronics"
(S. Chand) - chapters 1 (measurement & control systems), 3 (sensors &
transducers), 4 (signal conditioning / data acquisition), 5 (microprocessors),
6 (system models & controllers), 7 (actuators), 8 (mechatronic systems) and
9 (CNC).
"""
from __future__ import annotations

from collections import deque
from typing import Iterable


# ---------------------------------------------------------------------------
# 1. The 45 canonical topics (topologically ordered: prereqs always come first)
# ---------------------------------------------------------------------------
TOPICS: list[str] = [
    # --- Tier 0 : roots -----------------------------------------------------
    "Sensors and transducers",                 # 0
    "Basic electronics",                       # 1
    "Operational amplifiers",                  # 2
    "Open-loop and closed-loop control",       # 3
    "Number systems and digital coding",       # 4
    # --- Tier 1 -------------------------------------------------------------
    "Sensor characteristics and performance",  # 5
    "Logic gates and Boolean algebra",         # 6
    "Signal conditioning",                     # 7
    "Control systems",                         # 8
    "Measurement systems",                     # 9
    # --- Tier 2 -------------------------------------------------------------
    "Position and displacement sensors",       # 10
    "Temperature sensors",                     # 11
    "Strain gauges",                           # 12
    "Sensor errors and calibration",           # 13
    "Filtering",                               # 14
    "Operational amplifier signal conditioning",  # 15
    "Analog-to-digital conversion",            # 16
    "Flip-flops and registers",                # 17
    "System transfer functions",               # 18
    "Block diagram and signal flow",           # 19
    # --- Tier 3 -------------------------------------------------------------
    "Optical encoders",                        # 20
    "Velocity and acceleration sensors",       # 21
    "Thermocouples",                           # 22
    "Light and proximity sensors",             # 23
    "Data acquisition systems",                # 24
    "Digital input-output interfacing",        # 25
    "Counters and timers",                     # 26
    "Microprocessor architecture",             # 27
    "Dynamic system response",                 # 28
    "Frequency response",                      # 29
    # --- Tier 4 -------------------------------------------------------------
    "Microcontroller interfacing",             # 30
    "Electrical actuation",                    # 31
    "Hydraulic actuation",                     # 32
    "Pneumatic actuation",                     # 33
    "Stepper and servo drives",                # 34
    "Controllers",                             # 35
    "Closed-loop dynamic response",            # 36
    # --- Tier 5 -------------------------------------------------------------
    "PID control",                             # 37
    "Closed-loop stability",                   # 38
    "Programmable logic controllers",          # 39
    "Digital control",                         # 40
    # --- Tier 6 -------------------------------------------------------------
    "Sequential and process control",          # 41
    "Fault diagnosis and safety logic",        # 42
    "Mechatronic system design",               # 43
    "Mechatronic system integration",          # 44
]

N_TOPICS = len(TOPICS)
TOPIC_TO_IDX = {topic: i for i, topic in enumerate(TOPICS)}


# ---------------------------------------------------------------------------
# 2. Topic clusters (chapter-style groups). Used by the "blocked" learner who
#    benefits from working through a cluster before moving to the next one.
# ---------------------------------------------------------------------------
TOPIC_CLUSTERS: dict[str, list[str]] = {
    "Foundations": [
        "Sensors and transducers",
        "Basic electronics",
        "Operational amplifiers",
        "Open-loop and closed-loop control",
        "Number systems and digital coding",
    ],
    "Sensing": [
        "Sensor characteristics and performance",
        "Measurement systems",
        "Position and displacement sensors",
        "Temperature sensors",
        "Strain gauges",
        "Sensor errors and calibration",
        "Optical encoders",
        "Velocity and acceleration sensors",
        "Thermocouples",
        "Light and proximity sensors",
    ],
    "Signal & data": [
        "Signal conditioning",
        "Filtering",
        "Operational amplifier signal conditioning",
        "Analog-to-digital conversion",
        "Data acquisition systems",
    ],
    "Digital & processors": [
        "Logic gates and Boolean algebra",
        "Flip-flops and registers",
        "Digital input-output interfacing",
        "Counters and timers",
        "Microprocessor architecture",
        "Microcontroller interfacing",
    ],
    "Actuation": [
        "Electrical actuation",
        "Hydraulic actuation",
        "Pneumatic actuation",
        "Stepper and servo drives",
    ],
    "Systems & control": [
        "Control systems",
        "System transfer functions",
        "Block diagram and signal flow",
        "Dynamic system response",
        "Frequency response",
        "Controllers",
        "Closed-loop dynamic response",
        "PID control",
        "Closed-loop stability",
        "Digital control",
    ],
    "Industrial & integration": [
        "Programmable logic controllers",
        "Sequential and process control",
        "Fault diagnosis and safety logic",
        "Mechatronic system design",
        "Mechatronic system integration",
    ],
}

# Reverse lookup: topic -> cluster name
TOPIC_TO_CLUSTER: dict[str, str] = {
    topic: cluster
    for cluster, members in TOPIC_CLUSTERS.items()
    for topic in members
}


# ---------------------------------------------------------------------------
# 3. Prerequisite DAG.  topic -> list of topics that should come first.
#    Every listed prerequisite appears earlier in TOPICS (=> acyclic).
# ---------------------------------------------------------------------------
PREREQUISITES: dict[str, list[str]] = {
    # Tier 0 roots have no prerequisites.
    "Sensors and transducers": [],
    "Basic electronics": [],
    "Operational amplifiers": ["Basic electronics"],
    "Open-loop and closed-loop control": [],
    "Number systems and digital coding": [],
    # Tier 1
    "Sensor characteristics and performance": ["Sensors and transducers"],
    "Logic gates and Boolean algebra": ["Number systems and digital coding", "Basic electronics"],
    "Signal conditioning": ["Operational amplifiers"],
    "Control systems": ["Open-loop and closed-loop control"],
    "Measurement systems": ["Sensors and transducers"],
    # Tier 2
    "Position and displacement sensors": ["Sensors and transducers", "Sensor characteristics and performance"],
    "Temperature sensors": ["Sensors and transducers", "Sensor characteristics and performance"],
    "Strain gauges": ["Sensors and transducers", "Sensor characteristics and performance"],
    "Sensor errors and calibration": ["Sensor characteristics and performance", "Measurement systems"],
    "Filtering": ["Signal conditioning"],
    "Operational amplifier signal conditioning": ["Operational amplifiers", "Signal conditioning"],
    "Analog-to-digital conversion": ["Signal conditioning"],
    "Flip-flops and registers": ["Logic gates and Boolean algebra"],
    "System transfer functions": ["Control systems"],
    "Block diagram and signal flow": ["Control systems"],
    # Tier 3
    "Optical encoders": ["Position and displacement sensors"],
    "Velocity and acceleration sensors": ["Position and displacement sensors"],
    "Thermocouples": ["Temperature sensors"],
    "Light and proximity sensors": ["Sensors and transducers", "Sensor characteristics and performance"],
    "Data acquisition systems": ["Analog-to-digital conversion", "Signal conditioning"],
    "Digital input-output interfacing": ["Analog-to-digital conversion", "Flip-flops and registers"],
    "Counters and timers": ["Flip-flops and registers"],
    "Microprocessor architecture": ["Flip-flops and registers", "Number systems and digital coding"],
    "Dynamic system response": ["System transfer functions"],
    "Frequency response": ["System transfer functions"],
    # Tier 4
    "Microcontroller interfacing": ["Microprocessor architecture", "Digital input-output interfacing"],
    "Electrical actuation": ["Open-loop and closed-loop control", "Basic electronics"],
    "Hydraulic actuation": ["Open-loop and closed-loop control"],
    "Pneumatic actuation": ["Open-loop and closed-loop control"],
    "Stepper and servo drives": ["Electrical actuation", "Optical encoders"],
    "Controllers": ["Control systems", "Dynamic system response"],
    "Closed-loop dynamic response": ["Dynamic system response", "Controllers"],
    # Tier 5
    "PID control": ["Controllers"],
    "Closed-loop stability": ["Frequency response", "Closed-loop dynamic response"],
    "Programmable logic controllers": ["Digital input-output interfacing", "Logic gates and Boolean algebra"],
    "Digital control": ["Analog-to-digital conversion", "Controllers"],
    # Tier 6
    "Sequential and process control": ["Programmable logic controllers"],
    "Fault diagnosis and safety logic": ["Programmable logic controllers", "Sensor errors and calibration"],
    "Mechatronic system design": ["Controllers", "Electrical actuation", "Data acquisition systems"],
    "Mechatronic system integration": [
        "Mechatronic system design",
        "Programmable logic controllers",
        "Stepper and servo drives",
    ],
}


# ---------------------------------------------------------------------------
# 4. Transfer / interference matrix.  (A, B) -> delta applied to topic B's
#    mastery when the student studies topic A.  + = transfer, - = interference.
# ---------------------------------------------------------------------------
TRANSFER_MATRIX: dict[tuple[str, str], float] = {
    # ---- positive transfer ------------------------------------------------
    ("Sensors and transducers", "Measurement systems"): +0.30,
    ("Sensor characteristics and performance", "Sensor errors and calibration"): +0.35,
    ("Operational amplifiers", "Operational amplifier signal conditioning"): +0.40,
    ("Operational amplifiers", "Signal conditioning"): +0.30,
    ("Signal conditioning", "Filtering"): +0.30,
    ("Signal conditioning", "Data acquisition systems"): +0.25,
    ("Analog-to-digital conversion", "Data acquisition systems"): +0.35,
    ("Position and displacement sensors", "Optical encoders"): +0.30,
    ("Position and displacement sensors", "Velocity and acceleration sensors"): +0.25,
    ("Temperature sensors", "Thermocouples"): +0.40,
    ("Number systems and digital coding", "Logic gates and Boolean algebra"): +0.35,
    ("Logic gates and Boolean algebra", "Flip-flops and registers"): +0.35,
    ("Flip-flops and registers", "Counters and timers"): +0.35,
    ("Flip-flops and registers", "Digital input-output interfacing"): +0.25,
    ("Microprocessor architecture", "Microcontroller interfacing"): +0.40,
    ("Control systems", "System transfer functions"): +0.30,
    ("System transfer functions", "Block diagram and signal flow"): +0.30,
    ("System transfer functions", "Dynamic system response"): +0.30,
    ("Dynamic system response", "Frequency response"): +0.30,
    ("Dynamic system response", "Closed-loop dynamic response"): +0.30,
    ("Controllers", "PID control"): +0.45,
    ("PID control", "Closed-loop stability"): +0.25,
    ("Frequency response", "Closed-loop stability"): +0.30,
    ("Electrical actuation", "Stepper and servo drives"): +0.35,
    ("Programmable logic controllers", "Sequential and process control"): +0.40,
    ("Hydraulic actuation", "Pneumatic actuation"): +0.20,
    ("Controllers", "Closed-loop dynamic response"): +0.30,
    ("Open-loop and closed-loop control", "Control systems"): +0.30,
    ("Mechatronic system design", "Mechatronic system integration"): +0.35,
    # ---- negative interference -------------------------------------------
    ("PID control", "Open-loop and closed-loop control"): -0.10,
    ("Pneumatic actuation", "Hydraulic actuation"): -0.10,
    ("Digital control", "PID control"): -0.10,
    ("Stepper and servo drives", "Electrical actuation"): -0.08,
    ("Frequency response", "Dynamic system response"): -0.08,
    ("Fault diagnosis and safety logic", "Sequential and process control"): -0.08,
}


# ---------------------------------------------------------------------------
# 5. Helpers
# ---------------------------------------------------------------------------
# Global multiplier on every transfer/interference edge.  Raised above 1.0 so the
# *order* in which topics are unlocked compounds: teaching a hub topic well now
# meaningfully boosts its downstream targets, making a smart curriculum order far
# better than a greedy one (extra headroom for RL over fixed rules).
TRANSFER_SCALE: float = 1.8


def transfer_targets(source_topic: str) -> dict[str, float]:
    """All (target_topic -> delta) entries spilling out of ``source_topic``."""
    return {
        b: delta * TRANSFER_SCALE
        for (a, b), delta in TRANSFER_MATRIX.items()
        if a == source_topic
    }


def prerequisites_of(topic: str) -> list[str]:
    return list(PREREQUISITES.get(topic, []))


def all_ancestors(topic: str) -> set[str]:
    """Transitive closure of prerequisites (everything that must precede topic)."""
    seen: set[str] = set()
    stack = list(PREREQUISITES.get(topic, []))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(PREREQUISITES.get(node, []))
    return seen


def topological_order() -> list[str]:
    """Kahn's algorithm -> raises if the graph is cyclic."""
    indeg = {t: 0 for t in TOPICS}
    children: dict[str, list[str]] = {t: [] for t in TOPICS}
    for topic, prereqs in PREREQUISITES.items():
        for pre in prereqs:
            children[pre].append(topic)
            indeg[topic] += 1
    queue = deque([t for t in TOPICS if indeg[t] == 0])
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(order) != len(TOPICS):
        raise ValueError("PREREQUISITES graph contains a cycle!")
    return order


def validate_dag(strict: bool = True) -> dict[str, object]:
    """Validate the curriculum graph; returns a small report dict."""
    problems: list[str] = []

    # every PREREQUISITES key + value must be a known topic
    for topic, prereqs in PREREQUISITES.items():
        if topic not in TOPIC_TO_IDX:
            problems.append(f"unknown topic in PREREQUISITES key: {topic!r}")
        for pre in prereqs:
            if pre not in TOPIC_TO_IDX:
                problems.append(f"unknown prerequisite {pre!r} for {topic!r}")
            if pre == topic:
                problems.append(f"self-loop on {topic!r}")

    # every topic must appear as a key
    for topic in TOPICS:
        if topic not in PREREQUISITES:
            problems.append(f"topic missing from PREREQUISITES: {topic!r}")

    # transfer matrix keys must be known topics
    for (a, b) in TRANSFER_MATRIX:
        if a not in TOPIC_TO_IDX or b not in TOPIC_TO_IDX:
            problems.append(f"unknown topic in TRANSFER_MATRIX pair: ({a!r}, {b!r})")

    # cluster membership must be a partition of TOPICS
    clustered = [t for members in TOPIC_CLUSTERS.values() for t in members]
    if sorted(clustered) != sorted(TOPICS):
        problems.append("TOPIC_CLUSTERS is not a clean partition of TOPICS")

    # acyclicity
    cyclic = False
    try:
        topological_order()
    except ValueError:
        cyclic = True
        problems.append("graph is cyclic")

    if strict and problems:
        raise ValueError("Curriculum validation failed:\n  - " + "\n  - ".join(problems))

    return {
        "n_topics": N_TOPICS,
        "n_prereq_edges": sum(len(v) for v in PREREQUISITES.values()),
        "n_transfer_edges": len(TRANSFER_MATRIX),
        "n_clusters": len(TOPIC_CLUSTERS),
        "acyclic": not cyclic,
        "problems": problems,
    }


if __name__ == "__main__":
    report = validate_dag(strict=True)
    print("Curriculum OK")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("\nTopological order (first 10):")
    for t in topological_order()[:10]:
        print(f"  - {t}")

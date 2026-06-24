# 🎓 Adaptive MCQ Learning System with RL & LLM

An intelligent tutoring system that uses **Reinforcement Learning (RL)** and **Large Language Models (LLM)** to adaptively personalize multiple-choice question (MCQ) learning sessions for mechatronics education.

## 📁 Repository Contents

This repository contains **two independent but related projects**:

### 1. **Student_SessionLSTM_Based_RL_&_grok_api** — Main Research System
The primary adaptive learning platform with:
- **5 RL algorithms** (RecurrentPPO, A2C+LSTM, DRQN+LSTM, and legacy baselines)
- **675 pre-curated MCQs** (45 topics × 15 questions each)
- **LLM integration** (Groq API for explanations, question generation, learning paths)
- **Explainable AI** (XAI) showing why topics are selected
- **Student persistence** across sessions with ability tracking
- **Flask web UI** for interactive sessions

📖 **Full documentation**: See [`Student_SessionLSTM_Based_RL_&_grok_api/README.md`](Student_SessionLSTM_Based_RL_&_grok_api/README.md)

### 2. **Real_Student_Session_Lab_demo** — Demo Application
A simplified, deployment-ready version:
- Live interactive web interface
- Pre-trained models included
- Streamlined UI for end users
- Designed for Render.com / HuggingFace deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Groq API key (free tier: https://console.groq.com)
- Ollama (for local embeddings)

### Setup & Run
```bash
cd Student_SessionLSTM_Based_RL_&_grok_api

# Install dependencies
pip install -r requirements.txt

# Set Groq API key
export GROQ_API_KEY="gsk_your_key_here"

# Build embeddings cache
python retriever.py --build

# Run interactive session
python run_real_student_session.py \
    --model runs/double_dqn_lstm/models/best_model.pt \
    --student-id alice
```

## 🌟 Key Features

- **Adaptive Difficulty**: Delta-driven controller matches question difficulty to student ability
- **Memory-Equipped RL**: LSTM-based agents learn optimal topic sequencing
- **LLM Explanations**: Groq API provides grounded, contextual explanations
- **Cross-Session Learning**: Student history persists across multiple sessions
- **Curriculum Awareness**: Anti-massing, coverage carryover, and learning paths
- **Explainability**: Occlusion-based XAI shows why the agent picks each topic
- **Grounded Generation**: Fallback to LLM-generated MCQs when the bank doesn't fit
- **Offline-Ready**: Embeddings via local Ollama (`nomic-embed-text`)

## 📊 How It Works

1. **Environment** (`mcq_env.py`): Gym-compatible RL environment tracking student ability and learning
2. **RL Training**: 5 different algorithms compete to learn optimal topic selection
3. **Real Sessions**: Agents interact with real students, adapting difficulty in real-time
4. **Evaluation**: Compare trained agents against baselines using true mastery metrics

## 📂 Folder Structure

```
.
├── Student_SessionLSTM_Based_RL_&_grok_api/     Main system (all code, research)
│   ├── app.py                                    Flask web UI
│   ├── mcq_env.py                               RL environment
│   ├── train_*.py                               5 RL trainers
│   ├── run_real_student_session.py              Interactive session runner
│   ├── evaluate_baselines.py                    Scoreboard comparison
│   └── README.md                                Full documentation
│
├── Real_Student_Session_Lab_demo/               Demo application
│   ├── app.py                                   Web UI
│   ├── render.yaml                              Deployment config
│   └── requirements.txt                         Dependencies
│
└── README.md                                     This file
```

## 🔧 Main Tools & Components

| Component | Purpose | Language |
|-----------|---------|----------|
| **RL Trainers** | Train agents to pick optimal topics | Python + PyTorch |
| **Session Runner** | Run interactive/simulated student sessions | Python |
| **LLM Client** | Groq API integration for generation | Python |
| **Retriever** | Ollama embeddings + PDF retrieval | Python |
| **Web UI** | Flask-based interactive interface | Python + HTML/CSS/JS |

## 📚 Key Papers & Concepts

- **Adaptive Difficulty**: Delta-driven controller (inspired by IRT)
- **RL Algorithms**: PPO, A2C, Double DQN with LSTM memory
- **Student Modeling**: Continuous ability, Elo-style updates, spacing/forgetting
- **Curriculum**: Topic sequencing via RL, coverage carryover, anti-massing
- **Explainability**: Occlusion-based counterfactuals for policy decisions

## 🤝 Contributing

For research improvements, model additions, or new evaluation metrics:
1. Check [`Student_SessionLSTM_Based_RL_&_grok_api/PLAN.md`](Student_SessionLSTM_Based_RL_&_grok_api/PLAN.md) for architecture
2. Follow existing code patterns (torch checkpoints, gym env interface, CSV logging)
3. Add baseline comparisons in `final_scoreboard.py`

## 📄 License

This project is for educational and research purposes.

## 🙋 Questions?

Refer to the detailed README in [`Student_SessionLSTM_Based_RL_&_grok_api/README.md`](Student_SessionLSTM_Based_RL_&_grok_api/README.md) for:
- Complete setup instructions
- Training commands for all 5 algorithms
- Running sessions (interactive, simulated, or batch)
- Evaluation and scoreboard creation
- Troubleshooting and verification checklists

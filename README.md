# Wireless QoS Security: Network Architecture & QoS Assurance through Secure AI Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Gymnasium-orange.svg)]()

## 📋 Project Overview

This project implements a comprehensive simulation framework for the PhD research topic: **"Network Architecture and Quality of Service Assurance through Secure AI Models."**

As 5G/6G networks increasingly rely on Deep Reinforcement Learning (DRL) for critical resource allocation, they become susceptible to adversarial attacks. This framework simulates a realistic wireless environment to study how adversarial attacks (specifically **CQI Falsification** and **Black Hole attacks**) impact network Quality of Service (QoS). Furthermore, it implements and evaluates **Robust AI defense mechanisms** to ensure network resilience and fairness.

### 🔑 Key Components
* **📡 5G/6G Network Simulator:** A Gymnasium-based environment simulating multiple users, Resource Blocks (RBs), realistic channel models (Path loss, Fading, SINR), and traffic dynamics.
* **🤖 DRL Agent:** A Proximal Policy Optimization (PPO) agent responsible for dynamic resource allocation.
* **⚔️ Attack Module:** Implements adversarial attacks, focusing on **"Black Hole"** strategies where malicious users report falsified perfect Channel Quality Indicators (CQI=1.0) to starve legitimate users of resources.
* **🛡️ Defense Module:** A multi-layered defense suite including Adversarial Training, Statistical Anomaly Detection, and Input Validation.
* **📊 Evaluation Framework:** Comprehensive metrics for Throughput, Latency, Jain's Fairness Index, and Security resilience.

---

## 📁 Project Structure

```text
wireless_qos_security/
├── environment/
│   ├── network_env.py       # 5G/6G network simulator (Gym environment with Attack logic)
│   ├── channel_model.py     # Physical layer (path loss, fading, SINR, CQI mapping)
│   └── qos_metrics.py       # QoS calculators (throughput, latency, fairness)
├── agents/
│   ├── ppo_agent.py         # Standard PPO implementation (Stable-Baselines3)
│   └── robust_ppo.py        # Robust PPO agent wrapper with integrated defenses
├── attacks/
│   ├── cqi_poisoning.py     # CQI falsification logic
│   └── attack_strategies.py # Definitions of various attack vectors
├── defenses/
│   ├── adversarial_training.py # Wrapper for training on poisoned data
│   ├── anomaly_detector.py     # Statistical anomaly detection (Z-score, IQR)
│   └── input_validator.py      # Input sanitization and rate limiting
├── configs/
│   └── config.yaml          # Centralized hyperparameter configuration
├── results/                 # Output directory for logs, models, and figures
├── train.py                 # Main training script
├── evaluate.py              # Evaluation & comparison script
├── plot_results.py          # Visualization script for publication figures
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

⚙️ Installation
1. Clone the Repository

Bash
git clone [https://github.com/korinaak/wireless_qos_security.git](https://github.com/korinaak/wireless_qos_security.git)
cd wireless_qos_security
2. Create a Virtual Environment

It is recommended to use a clean virtual environment to avoid dependency conflicts.

Bash
# Create environment
python3 -m venv drl-env

# Activate environment
# On Linux/Mac:
source drl-env/bin/activate
# On Windows:
# drl-env\Scripts\activate
3. Install Dependencies

Bash
pip install -r requirements.txt
🚀 Usage
1. Configuration

Modify configs/config.yaml to customize:

Network: Number of users, Resource Blocks (RBs), max steps.

Attack: Probability of malicious users, attack magnitude.

PPO: Learning rate, batch size, gamma.

2. Training

You can train the Baseline, Attacked, and Robust agents sequentially.

Train all scenarios:

Bash
python train.py --config configs/config.yaml --scenarios all
Train specific scenarios:

Bash
# Train only baseline (Clean environment)
python train.py --scenarios baseline

# Train baseline and robust defense agent
python train.py --scenarios baseline robust
3. Evaluation

Evaluate the trained models to generate JSON metrics.

Bash
python evaluate.py \
  --config configs/config.yaml \
  --model_dir results/training_logs \
  --n_episodes 50 \
  --output results/evaluation_results.json
4. Visualization

Generate publication-quality plots (Bar charts for Fairness, Latency, Throughput).

Bash
python plot_results.py \
  --results results/evaluation_results.json \
  --output_dir results/figures
Outputs:

fairness_comparison.png

latency_comparison.png

throughput_comparison.png

latex_table.tex (Auto-generated code for LaTeX papers)

📊 Experiment Scenarios
The framework evaluates the system under three distinct conditions:

Scenario	Description	Environment Condition	Agent Type	Goal
1. Baseline	Ideal Operation	Clean (0% Malicious)	Standard PPO	Establish upper bound performance benchmarks.
2. Under Attack	Adversarial	Poisoned (e.g., 30% Malicious)	Standard PPO	Demonstrate the vulnerability of standard AI agents to Black Hole attacks (Malicious users reporting max CQI to steal resources).
3. Robust AI	Defense Enabled	Poisoned (e.g., 30% Malicious)	Robust PPO	Demonstrate QoS recovery using defense mechanisms (Adversarial Training, Anomaly Detection).
🛡️ Defense Mechanisms
Adversarial Training:

The agent is trained in an environment that naturally contains adversarial perturbations.

It learns a policy that identifies "lying" agents (high reported CQI but low resulting reward) and ignores them.

Anomaly Detection:

Uses statistical methods (Z-score, Moving Average) to flag CQI reports that deviate significantly from historical patterns or physical constraints.

Input Validation:

Sanitizes inputs by enforcing bounds and rate limits to prevent sudden, unrealistic jumps in channel quality reporting.

🔬 Key Metrics
Quality of Service (QoS)

Throughput: Aggregate network capacity (Mbps).

Latency: Average queueing delay per user (ms).

Jain's Fairness Index: Measures how equally resources are distributed (Scale 0 to 1, where 1 is perfect fairness).

Security

Attack Impact: The degradation percentage in Fairness and Throughput relative to the baseline.

Defense Recovery: The percentage of QoS performance restored by the Robust Agent compared to the Attacked scenario.

📈 Experimental Results
Based on simulation runs using Black Hole attacks, the framework demonstrates the following trends:

Metric	Baseline (Clean)	Under Attack	Robust Defense	Analysis
Fairness	High (~0.96)	Critical Drop (~0.25)	Recovering (~0.35+)	Attacks cause severe starvation of legitimate users.
Latency	Low (~23ms)	High (>600ms)	Stabilized	Attackers clog the network; Defense allows queues to drain.
Throughput	~22 Mbps	~21.5 Mbps	~21.6 Mbps	Aggregate throughput remains similar, but distribution becomes highly inequitable under attack.
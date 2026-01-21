Wireless QoS Security: Network Architecture and QoS Assurance through Secure AI Models
📋 Project Overview
This project implements a comprehensive simulation framework for studying adversarial attacks on AI-driven resource allocation in 5G/6G wireless networks and evaluating robust defense mechanisms to maintain Quality of Service (QoS).
Key Components:
Network Simulator: Realistic 5G/6G environment with multiple users, Resource Blocks (RBs), and channel models
DRL Agent: PPO-based resource allocator
Attack Module: CQI falsification and other adversarial attacks
Defense Module: Adversarial training, anomaly detection, input validation
Evaluation Framework: Comprehensive QoS metrics and visualization
📁 Project Structure
wireless_qos_security/
├── environment/
│   ├── network_env.py          # 5G/6G network simulator (Gym environment)
│   ├── channel_model.py        # Physical layer (path loss, fading, SINR)
│   └── qos_metrics.py          # QoS calculators (throughput, latency, fairness)
├── agents/
│   ├── ppo_agent.py            # Standard PPO implementation
│   └── robust_ppo.py           # Robust PPO with defenses
├── attacks/
│   ├── cqi_poisoning.py        # CQI falsification attack
│   └── attack_strategies.py    # Various attack types
├── defenses/
│   ├── adversarial_training.py # Adversarial training defense
│   ├── anomaly_detector.py     # Statistical anomaly detection
│   └── input_validator.py      # Input sanitization
├── configs/
│   └── config.yaml             # Centralized configuration
├── results/                    # Output folder (logs, models, figures)
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation & comparison
├── plot_results.py             # Visualization for paper
├── requirements.txt            # Dependencies
└── README.md                   # This file
1. Installation
bash
# Clone repository
git clone <your-repo-url>
cd wireless_qos_security

# Create virtual environment
python3 -m venv drl-env
To activate drl-env:
source drl-env/bin/activate
To exit: 
deactivate

# Install dependencies
pip install -r requirements.txt
2. Configuration
Edit configs/config.yaml to customize:
Network parameters (users, RBs, bandwidth)
PPO hyperparameters
Attack configuration
Defense mechanisms
3. Training
Train all three scenarios (baseline, under attack, robust):
bash
python train.py --config configs/config.yaml --scenarios all
Or train specific scenarios:
bash
# Train only baseline
python train.py --scenarios baseline

# Train baseline and robust
python train.py --scenarios baseline robust
4. Evaluation
Evaluate trained models:
bash
python evaluate.py \
    --config configs/config.yaml \
    --model_dir results/training_<timestamp> \
    --n_episodes 50 \
    --output results/evaluation_results.json
5. Generate Plots
Create publication-quality figures:
bash
python plot_results.py \
    --results results/evaluation_results.json \
    --output_dir results/figures
Outputs:
performance_comparison.png/pdf
attack_impact.png/pdf
defense_effectiveness.png/pdf
detailed_metrics.png/pdf
latex_table.tex (for paper)
📊 Experiment Scenarios
Scenario 1: Baseline (No Attack)
Environment: Clean, no adversarial users
Agent: Standard PPO
Purpose: Establish upper bound on performance
Scenario 2: Under Attack (No Defense)
Environment: 30% malicious users (CQI falsification)
Agent: Standard PPO (trained on clean data)
Purpose: Demonstrate vulnerability of ML-based systems
Scenario 3: Robust AI Defense
Environment: 30% malicious users
Agent: Robust PPO with:
Adversarial training (20% poisoned samples during training)
Anomaly detection (Z-score based)
Input validation (bounds checking, rate limiting)
Purpose: Show effectiveness of defense mechanisms
🔬 Key Metrics
The framework evaluates:
QoS Metrics:
Throughput: Total network capacity (Mbps)
Latency: Average queueing delay (ms)
Fairness: Jain's Fairness Index [0, 1]
User Satisfaction: % of users meeting QoS requirements
5th Percentile Throughput: Cell-edge performance
Security Metrics:
Attack Success Rate: Impact on QoS
Detection Rate: % of anomalies detected
Defense Recovery: QoS restoration percentage
🛡️ Defense Mechanisms
1. Adversarial Training
Trains PPO on mixture of clean and poisoned samples
Builds inherent robustness to perturbations
Configured via defense.adversarial_training in config
2. Anomaly Detection
Statistical outlier detection (Z-score, IQR, Moving Average)
Detects abnormal CQI reports
Can use ensemble of detectors for higher accuracy
3. Input Validation
Bounds checking (clip to valid ranges)
Rate limiting (prevent sudden jumps)
Consistency checking (cross-validate CQI and buffer state)
🎯 Expected Results
Based on literature and preliminary experiments:
Scenario	Throughput	Fairness	Latency
Baseline (Clean)	100%	0.85-0.90	Low
Under Attack	60-70% ↓	0.60-0.70 ↓	High ↑
With Defense	85-95% ↑	0.80-0.85 ↑	Medium
Attack Degradation: ~30% throughput loss, ~25% fairness reduction
Defense Recovery: ~80% QoS restoration


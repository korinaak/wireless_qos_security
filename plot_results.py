"""
Plot Results for Paper

Generates publication-quality figures:
1. Performance comparison (throughput, latency, fairness)
2. Attack impact visualization
3. Defense effectiveness
4. Box plots and error bars
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300


def load_results(results_path: str):
    """Load evaluation results from JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_performance_comparison(results, output_dir):
    """
    Create bar plots comparing throughput, fairness, and latency
    """
    scenarios = {
        'baseline_clean': 'Baseline\n(No Attack)',
        'trained_under_attack': 'Baseline\n(Under Attack)',
        'robust_attack': 'Robust PPO\n(Under Attack)'
    }
    
    metrics = ['mean_throughput', 'mean_fairness', 'mean_latency']
    metric_names = ['Throughput (Mbps)', 'Jain\'s Fairness Index', 'Average Latency (ms)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        scenario_names = []
        values = []
        errors = []
        
        for key, name in scenarios.items():
            if key in results:
                scenario_names.append(name)
                values.append(results[key][metric])
                # Error bars (std dev)
                std_key = metric.replace('mean_', 'std_')
                errors.append(results[key].get(std_key, 0))
        
        # Colors
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        bars = ax.bar(range(len(values)), values, yerr=errors, 
                     color=colors[:len(values)], alpha=0.8, capsize=5)
        
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(scenario_names)
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'performance_comparison.pdf'), bbox_inches='tight')
    print(f"✓ Saved performance_comparison.png/pdf")
    plt.close()


def plot_attack_impact(results, output_dir):
    """
    Visualize the impact of attacks
    """
    if 'baseline_clean' not in results or 'trained_under_attack' not in results:
        print("Skipping attack impact plot (missing data)")
        return
    
    baseline = results['baseline_clean']
    attacked = results['trained_under_attack']
    
    metrics = ['mean_throughput', 'mean_fairness', 'mean_latency']
    metric_names = ['Throughput\n(Mbps)', 'Fairness\nIndex', 'Latency\n(ms)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_values = [baseline[m] for m in metrics]
    attacked_values = [attacked[m] for m in metrics]
    
    # Normalize to show relative change
    normalized_baseline = [100] * len(metrics)
    normalized_attacked = [(attacked_values[i] / baseline_values[i]) * 100 
                           for i in range(len(metrics))]
    
    bars1 = ax.bar(x - width/2, normalized_baseline, width, label='Clean (Baseline)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, normalized_attacked, width, label='Under Attack', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Relative Performance (%)')
    ax.set_title('Attack Impact on QoS Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_impact.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'attack_impact.pdf'), bbox_inches='tight')
    print(f"✓ Saved attack_impact.png/pdf")
    plt.close()


def plot_defense_effectiveness(results, output_dir):
    """
    Show defense recovery performance
    """
    if 'trained_under_attack' not in results or 'robust_attack' not in results:
        print("Skipping defense effectiveness plot (missing data)")
        return
    
    attacked = results['trained_under_attack']
    defended = results['robust_attack']
    baseline = results.get('baseline_clean', attacked)
    
    metrics = ['mean_throughput', 'mean_fairness', 'mean_latency']
    metric_names = ['Throughput', 'Fairness', 'Latency']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    baseline_values = [baseline[m] for m in metrics]
    attacked_values = [attacked[m] for m in metrics]
    defended_values = [defended[m] for m in metrics]
    
    # Normalize
    norm_baseline = [100] * len(metrics)
    norm_attacked = [(attacked_values[i] / baseline_values[i]) * 100 for i in range(len(metrics))]
    norm_defended = [(defended_values[i] / baseline_values[i]) * 100 for i in range(len(metrics))]
    
    bars1 = ax.bar(x - width, norm_baseline, width, label='Baseline (Clean)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, norm_attacked, width, label='Under Attack (No Defense)', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, norm_defended, width, label='With Robust AI Defense', color='#3498db', alpha=0.8)
    
    ax.set_ylabel('Relative Performance (%)')
    ax.set_title('Defense Effectiveness: Recovery of QoS Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'defense_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'defense_effectiveness.pdf'), bbox_inches='tight')
    print(f"✓ Saved defense_effectiveness.png/pdf")
    plt.close()


def plot_detailed_metrics(results, output_dir):
    """
    Detailed plot with error bars for all metrics
    """
    scenarios_to_plot = [
        ('baseline_clean', 'Baseline\n(Clean)'),
        ('trained_under_attack', 'Baseline\n(Attack)'),
        ('robust_attack', 'Robust\n(Attack)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('mean_throughput', 'std_throughput', 'Total Throughput (Mbps)'),
        ('mean_fairness', 'std_fairness', 'Jain\'s Fairness Index'),
        ('mean_latency', 'std_latency', 'Average Latency (ms)'),
        ('mean_satisfaction', 'std_satisfaction', 'User Satisfaction Rate')
    ]
    
    for idx, (mean_key, std_key, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        scenario_names = []
        means = []
        stds = []
        
        for key, name in scenarios_to_plot:
            if key in results and mean_key in results[key]:
                scenario_names.append(name)
                means.append(results[key][mean_key])
                stds.append(results[key].get(std_key, 0))
        
        x_pos = np.arange(len(means))
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        ax.bar(x_pos, means, yerr=stds, color=colors[:len(means)], 
               alpha=0.8, capsize=5, width=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_names)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_metrics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'detailed_metrics.pdf'), bbox_inches='tight')
    print(f"✓ Saved detailed_metrics.png/pdf")
    plt.close()


def create_summary_table(results, output_dir):
    """
    Create a LaTeX table for the paper
    """
    table_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of QoS Metrics Across Scenarios}",
        "\\label{tab:qos_comparison}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Metric} & \\textbf{Baseline} & \\textbf{Under Attack} & \\textbf{Robust AI} \\\\",
        "& \\textbf{(Clean)} & \\textbf{(No Defense)} & \\textbf{(With Defense)} \\\\",
        "\\hline"
    ]
    
    if 'baseline_clean' in results and 'trained_under_attack' in results and 'robust_attack' in results:
        baseline = results['baseline_clean']
        attacked = results['trained_under_attack']
        robust = results['robust_attack']
        
        # Throughput
        table_lines.append(
            f"Throughput (Mbps) & {baseline['mean_throughput']:.2f} $\\pm$ {baseline.get('std_throughput', 0):.2f} & "
            f"{attacked['mean_throughput']:.2f} $\\pm$ {attacked.get('std_throughput', 0):.2f} & "
            f"{robust['mean_throughput']:.2f} $\\pm$ {robust.get('std_throughput', 0):.2f} \\\\"
        )
        
        # Fairness
        table_lines.append(
            f"Fairness Index & {baseline['mean_fairness']:.3f} $\\pm$ {baseline.get('std_fairness', 0):.3f} & "
            f"{attacked['mean_fairness']:.3f} $\\pm$ {attacked.get('std_fairness', 0):.3f} & "
            f"{robust['mean_fairness']:.3f} $\\pm$ {robust.get('std_fairness', 0):.3f} \\\\"
        )
        
        # Latency
        table_lines.append(
            f"Latency (ms) & {baseline['mean_latency']:.2f} $\\pm$ {baseline.get('std_latency', 0):.2f} & "
            f"{attacked['mean_latency']:.2f} $\\pm$ {attacked.get('std_latency', 0):.2f} & "
            f"{robust['mean_latency']:.2f} $\\pm$ {robust.get('std_latency', 0):.2f} \\\\"
        )
    
    table_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    table_path = os.path.join(output_dir, 'latex_table.tex')
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))
    
    print(f"✓ Saved LaTeX table to {table_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate plots for paper')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation results JSON')
    parser.add_argument('--output_dir', type=str, default='results/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    print("\nGenerating plots...")
    
    # Generate all plots
    plot_performance_comparison(results, args.output_dir)
    plot_attack_impact(results, args.output_dir)
    plot_defense_effectiveness(results, args.output_dir)
    plot_detailed_metrics(results, args.output_dir)
    
    # Generate LaTeX table
    create_summary_table(results, args.output_dir)
    
    print(f"\n{'='*70}")
    print("All figures generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print("  - performance_comparison.png/pdf")
    print("  - attack_impact.png/pdf")
    print("  - defense_effectiveness.png/pdf")
    print("  - detailed_metrics.png/pdf")
    print("  - latex_table.tex")
    print("\nYou can now include these figures in your paper!")


if __name__ == "__main__":
    main()

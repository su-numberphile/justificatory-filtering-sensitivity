"""
Justificatory Filtering Sensitivity Analysis

This script simulates and visualizes the conditions under which normative filtering of punishment
is more cost-effective than indiscriminate sanctioning. It implements a Monte Carlo analysis of 
the inequality (1 - q)(γ + β) > δ using randomized parameters and outputs summary statistics 
and diagnostic plots for the main paper.
"""

def main():
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Rectangle
        import matplotlib.patches as mpatches
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        print("Missing required libraries. Install them with:")
        print("pip install numpy pandas matplotlib seaborn scikit-learn")
        raise e

    # --- Settings ---
    np.random.seed(42)
    G = 5
    nD = 2
    samples = 5000

    # --- Sample Parameters ---
    q_vals = np.random.uniform(0.0, 1.0, samples)
    gamma_vals = np.random.uniform(0.1, 5.0, samples)
    beta_vals = np.random.uniform(0.1, 10.0, samples)
    delta_vals = np.random.uniform(0.01, 3.0, samples)

    # --- Compute Costs ---
    C_baseline = (G - 1) * nD * (gamma_vals + beta_vals)
    C_filter = q_vals * (G - 1) * nD * (gamma_vals + beta_vals) + (G - 1) * nD * delta_vals
    filtering_beneficial = C_filter < C_baseline

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'q': q_vals, 'gamma': gamma_vals, 'beta': beta_vals, 'delta': delta_vals,
        'C_baseline': C_baseline, 'C_filter': C_filter, 'filtering_beneficial': filtering_beneficial
    })
    df['lhs'] = (1 - df['q']) * (df['gamma'] + df['beta'])
    df['rhs'] = df['delta']
    df['savings'] = df['C_baseline'] - df['C_filter']
    pct_beneficial = (df['filtering_beneficial'].sum() / len(df)) * 100

    def compute_local_density(x, y, k_neighbors=50):
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        coords_norm = np.column_stack((x_norm, y_norm))
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(coords_norm))).fit(coords_norm)
        distances, _ = nbrs.kneighbors(coords_norm)
        return 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)

    # --- Plot 1 ---
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#d73027', '#4575b4']
    labels = ['Filtering Not Beneficial', 'Filtering Beneficial']

    for i, beneficial in enumerate([False, True]):
        mask = df['filtering_beneficial'] == beneficial
        subset_x, subset_y = df.loc[mask, 'lhs'].values, df.loc[mask, 'rhs'].values
        if len(subset_x) > 0:
            densities = compute_local_density(subset_x, subset_y, k_neighbors=30)
            base, max_size = 8, 60
            density_norm = (densities - densities.min()) / (densities.max() - densities.min()) if len(densities) > 1 else np.array([0])
            point_sizes = base + density_norm * (max_size - base)
            ax1.scatter(subset_x, subset_y, c=colors[i], alpha=0.6, s=point_sizes, label=labels[i])

    x_max, y_max = df['lhs'].max() * 1.05, df['rhs'].max()
    diag_max = max(x_max, y_max)
    ax1.plot([0, diag_max], [0, diag_max], 'k--', linewidth=2, alpha=0.8, label='Threshold: (1-q)(γ+β) = δ')
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, y_max)
    ax1.set_xlabel('Cost Saved by Withholding Unjustified Punishment\n(1-q)(γ+β)', fontsize=12)
    ax1.set_ylabel('Cognitive Cost of Evaluation (δ)', fontsize=12)
    ax1.set_title('Efficiency of Justificatory Filtering', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'Filtering beneficial when points\nfall below diagonal line:\n{pct_beneficial:.1f}% of cases favor filtering',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # --- Plot 2 ---
    beneficial_savings = df[df['filtering_beneficial']]['savings']
    not_beneficial_savings = df[~df['filtering_beneficial']]['savings']
    ax2.hist(beneficial_savings, bins=50, alpha=0.7, color=colors[1], label=f'Beneficial (n={len(beneficial_savings)})', density=True)
    ax2.hist(not_beneficial_savings, bins=50, alpha=0.7, color=colors[0], label=f'Not Beneficial (n={len(not_beneficial_savings)})', density=True)
    ax2.axvline(0, color='black', linestyle=':', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Cost Savings ($C_{baseline} - C_{filter}$)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of Cost Savings from Filtering', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f'Mean savings:\nBeneficial: {beneficial_savings.mean():.2f}\nNot beneficial: {not_beneficial_savings.mean():.2f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("filtering_efficiency_plot.png", dpi=300)
    plt.show()

    # --- Parameter Distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    params = ['q', 'gamma', 'beta', 'delta']
    labels = ['Proportion Punishable (q)', 'Punishment Cost (γ)', 'Cost to Punished (β)', 'Cognitive Cost (δ)']
    for i, (param, label) in enumerate(zip(params, labels)):
        ax = axes[i//2, i%2]
        ax.hist(df[df['filtering_beneficial']][param], bins=30, alpha=0.7, color=colors[1], label='Filtering Beneficial', density=True)
        ax.hist(df[~df['filtering_beneficial']][param], bins=30, alpha=0.7, color=colors[0], label='Filtering Not Beneficial', density=True)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Distribution of {label}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Parameter Distributions by Filtering Efficiency', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("parameter_distributions_by_outcome.png", dpi=300)
    plt.show()

    # --- Print Summary ---
    print("=== SENSITIVITY ANALYSIS SUMMARY ===")
    print(f"Total simulations: {samples:,}")
    print(f"Filtering beneficial in {pct_beneficial:.1f}% of cases")
    print("\nMean parameter values when filtering is beneficial:")
    for param, label in zip(params, labels):
        print(f"  {label}: {df[df['filtering_beneficial']][param].mean():.3f}")
    print("\nMean parameter values when filtering is NOT beneficial:")
    for param, label in zip(params, labels):
        print(f"  {label}: {df[~df['filtering_beneficial']][param].mean():.3f}")
    print("\nCost savings statistics:")
    print(f"  Mean savings when beneficial: {beneficial_savings.mean():.3f}")
    print(f"  Mean loss when not beneficial: {not_beneficial_savings.mean():.3f}")
    print(f"  Maximum savings: {df['savings'].max():.3f}")
    print(f"  Maximum loss: {df['savings'].min():.3f}")

    # --- Save CSV ---
    df.to_csv("sensitivity_analysis_output.csv", index=False)

if __name__ == "__main__":
    main()
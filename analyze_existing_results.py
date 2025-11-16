import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_all_results(results_dir='results'):
    """
    Load all CSV files from results directory
    """
    csv_files = glob.glob(os.path.join(results_dir, 'dqn_experiment_summary.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None
    
    print(f"Found {len(csv_files)} result files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and combine all CSVs
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data points loaded: {len(combined_df)}")
    print(f"Experiments: {combined_df['experiment'].unique()}")
    
    # Convert to list of dictionaries (same format as analyze_dqn_results expects)
    results = combined_df.to_dict('records')
    
    return results


def analyze_dqn_results(results):
    """
    Enhanced analysis with additional visualizations
    (Copy the entire function from my previous response here)
    """
    if len(results) == 0:
        print("No results to analyze")
        return None
    
    print(f"\nAnalyzing {len(results)} total data points...")
    
    df = pd.DataFrame(results)
    
    summary = df.groupby(['experiment', 'year']).agg({
        'gini': 'mean',
        'compliance_rate': 'mean',
        'evasion_rate': 'mean',
        'partial_payment_rate': 'mean',
        'total_sugar': 'mean',
        'population': 'mean',
        'epsilon': 'mean',
        'deaths': 'mean'
    }).reset_index()
    
    # Create output directory if it doesn't exist
    os.makedirs('analysis_output', exist_ok=True)
    
    # ==========================================
    # ORIGINAL 7 PLOTS
    # ==========================================
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('DQN Tax Enforcement Experiment Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Gini Coefficient
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 0].plot(exp_data.year, exp_data.gini, label=exp)
    axes[0, 0].set_title('Wealth Inequality (Gini Coefficient)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Gini Coefficient')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Partial Payment Rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 1].plot(exp_data.year, exp_data.partial_payment_rate, label=exp)
    axes[0, 1].set_title('Partial Payment Rate')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Partial Payment Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Population
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 0].plot(exp_data.year, exp_data.population, label=exp)
    axes[1, 0].set_title('Population Over Time')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Deaths
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 1].plot(exp_data.year, exp_data.deaths, label=exp)
    axes[1, 1].set_title('Agent Deaths per Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Deaths')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Epsilon
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[2, 0].plot(exp_data.year, exp_data.epsilon, label=exp)
    axes[2, 0].set_title('DQN Exploration Rate (Epsilon)')
    axes[2, 0].set_xlabel('Year')
    axes[2, 0].set_ylabel('Epsilon')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Evasion Rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[2, 1].plot(exp_data.year, exp_data.evasion_rate, label=exp)
    axes[2, 1].set_title('Tax Evasion Rate')
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].set_ylabel('Evasion Rate')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 7: Compliance Rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[3, 0].plot(exp_data.year, exp_data.compliance_rate, label=exp)
    axes[3, 0].set_title('Tax Compliance Rate')
    axes[3, 0].set_xlabel('Year')
    axes[3, 0].set_ylabel('Compliance Rate')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[3, 1])
    
    plt.tight_layout()
    plt.savefig('analysis_output/dqn_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # NEW VISUALIZATIONS
    # ==========================================
    
    # 1. LEARNING EFFICIENCY PLOT
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('DQN Learning Dynamics', fontsize=16, fontweight='bold')
    
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        
        # Epsilon vs Compliance
        axes2[0, 0].plot(exp_data.epsilon, exp_data.compliance_rate, 
                         label=exp, alpha=0.7, marker='o', markersize=2)
        axes2[0, 0].set_title('Compliance vs Exploration Rate')
        axes2[0, 0].set_xlabel('Epsilon (Exploration Rate)')
        axes2[0, 0].set_ylabel('Compliance Rate')
        axes2[0, 0].invert_xaxis()
        axes2[0, 0].legend(fontsize=8)
        axes2[0, 0].grid(True, alpha=0.3)
        
        # Epsilon vs Deaths
        axes2[0, 1].plot(exp_data.epsilon, exp_data.deaths, 
                         label=exp, alpha=0.7, marker='o', markersize=2)
        axes2[0, 1].set_title('Deaths vs Exploration Rate')
        axes2[0, 1].set_xlabel('Epsilon (Exploration Rate)')
        axes2[0, 1].set_ylabel('Deaths per Year')
        axes2[0, 1].invert_xaxis()
        axes2[0, 1].legend(fontsize=8)
        axes2[0, 1].grid(True, alpha=0.3)
        
        # Population retention
        axes2[1, 0].plot(exp_data.year, exp_data.population / exp_data.population.iloc[0], 
                         label=exp)
        axes2[1, 0].set_title('Population Retention Rate')
        axes2[1, 0].set_xlabel('Year')
        axes2[1, 0].set_ylabel('Population (Normalized to Start)')
        axes2[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        axes2[1, 0].legend(fontsize=8)
        axes2[1, 0].grid(True, alpha=0.3)
        
        # Cumulative deaths
        exp_data_copy = exp_data.copy()
        exp_data_copy['cumulative_deaths'] = exp_data_copy.deaths.cumsum()
        axes2[1, 1].plot(exp_data_copy.year, exp_data_copy.cumulative_deaths, label=exp)
        axes2[1, 1].set_title('Cumulative Deaths Over Time')
        axes2[1, 1].set_xlabel('Year')
        axes2[1, 1].set_ylabel('Total Deaths')
        axes2[1, 1].legend(fontsize=8)
        axes2[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_output/dqn_learning_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TAX BEHAVIOR PATTERNS
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Tax Compliance Behavior Patterns', fontsize=16, fontweight='bold')
    
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        
        # Compliance vs Evasion
        axes3[0, 0].plot(exp_data.compliance_rate, exp_data.evasion_rate, 
                         label=exp, alpha=0.6, marker='o', markersize=3)
        axes3[0, 0].set_title('Compliance vs Evasion Relationship')
        axes3[0, 0].set_xlabel('Compliance Rate')
        axes3[0, 0].set_ylabel('Evasion Rate')
        axes3[0, 0].legend(fontsize=8)
        axes3[0, 0].grid(True, alpha=0.3)
        
        # Inequality vs Compliance
        axes3[0, 1].plot(exp_data.gini, exp_data.compliance_rate, 
                         label=exp, alpha=0.6, marker='o', markersize=3)
        axes3[0, 1].set_title('Wealth Inequality vs Compliance')
        axes3[0, 1].set_xlabel('Gini Coefficient')
        axes3[0, 1].set_ylabel('Compliance Rate')
        axes3[0, 1].legend(fontsize=8)
        axes3[0, 1].grid(True, alpha=0.3)
        
        # Population vs Total Sugar
        axes3[1, 0].plot(exp_data.total_sugar, exp_data.population, 
                         label=exp, alpha=0.6, marker='o', markersize=3)
        axes3[1, 0].set_title('Population vs Total Sugar')
        axes3[1, 0].set_xlabel('Total Sugar in Economy')
        axes3[1, 0].set_ylabel('Population')
        axes3[1, 0].legend(fontsize=8)
        axes3[1, 0].grid(True, alpha=0.3)
        
        # Tax behavior stacked (just show first experiment as example)
        if exp == summary.experiment.unique()[0]:
            axes3[1, 1].fill_between(exp_data.year, 0, exp_data.compliance_rate, 
                                     alpha=0.5, label='Compliance', color='green')
            axes3[1, 1].fill_between(exp_data.year, exp_data.compliance_rate, 
                                     exp_data.compliance_rate + exp_data.partial_payment_rate,
                                     alpha=0.5, label='Partial', color='orange')
            axes3[1, 1].fill_between(exp_data.year, 
                                     exp_data.compliance_rate + exp_data.partial_payment_rate,
                                     exp_data.compliance_rate + exp_data.partial_payment_rate + exp_data.evasion_rate,
                                     alpha=0.5, label='Evasion', color='red')
            axes3[1, 1].set_title(f'Tax Behavior Composition - {exp}')
            axes3[1, 1].set_xlabel('Year')
            axes3[1, 1].set_ylabel('Rate')
            axes3[1, 1].legend()
            axes3[1, 1].grid(True, alpha=0.3)
            axes3[1, 1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('analysis_output/dqn_tax_behavior_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LENIENT VS STRICT COMPARISON
    lenient_exps = [e for e in summary.experiment.unique() if 'lenient' in e]
    strict_exps = [e for e in summary.experiment.unique() if 'strict' in e]
    
    if lenient_exps and strict_exps:
        fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
        fig4.suptitle('Lenient vs Strict Enforcement Comparison', fontsize=16, fontweight='bold')
        
        lenient_data = summary[summary.experiment.isin(lenient_exps)].groupby('year').mean()
        strict_data = summary[summary.experiment.isin(strict_exps)].groupby('year').mean()
        
        # Population
        axes4[0, 0].plot(lenient_data.index, lenient_data.population, 
                         label='Lenient', linewidth=2, color='green')
        axes4[0, 0].plot(strict_data.index, strict_data.population, 
                         label='Strict', linewidth=2, color='red')
        axes4[0, 0].set_title('Population: Lenient vs Strict')
        axes4[0, 0].set_xlabel('Year')
        axes4[0, 0].set_ylabel('Average Population')
        axes4[0, 0].legend()
        axes4[0, 0].grid(True, alpha=0.3)
        
        # Compliance
        axes4[0, 1].plot(lenient_data.index, lenient_data.compliance_rate, 
                         label='Lenient', linewidth=2, color='green')
        axes4[0, 1].plot(strict_data.index, strict_data.compliance_rate, 
                         label='Strict', linewidth=2, color='red')
        axes4[0, 1].set_title('Compliance: Lenient vs Strict')
        axes4[0, 1].set_xlabel('Year')
        axes4[0, 1].set_ylabel('Compliance Rate')
        axes4[0, 1].legend()
        axes4[0, 1].grid(True, alpha=0.3)
        
        # Gini
        axes4[0, 2].plot(lenient_data.index, lenient_data.gini, 
                         label='Lenient', linewidth=2, color='green')
        axes4[0, 2].plot(strict_data.index, strict_data.gini, 
                         label='Strict', linewidth=2, color='red')
        axes4[0, 2].set_title('Inequality: Lenient vs Strict')
        axes4[0, 2].set_xlabel('Year')
        axes4[0, 2].set_ylabel('Gini Coefficient')
        axes4[0, 2].legend()
        axes4[0, 2].grid(True, alpha=0.3)
        
        # Deaths
        axes4[1, 0].plot(lenient_data.index, lenient_data.deaths, 
                         label='Lenient', linewidth=2, color='green')
        axes4[1, 0].plot(strict_data.index, strict_data.deaths, 
                         label='Strict', linewidth=2, color='red')
        axes4[1, 0].set_title('Deaths: Lenient vs Strict')
        axes4[1, 0].set_xlabel('Year')
        axes4[1, 0].set_ylabel('Deaths per Year')
        axes4[1, 0].legend()
        axes4[1, 0].grid(True, alpha=0.3)
        
        # Evasion
        axes4[1, 1].plot(lenient_data.index, lenient_data.evasion_rate, 
                         label='Lenient', linewidth=2, color='green')
        axes4[1, 1].plot(strict_data.index, strict_data.evasion_rate, 
                         label='Strict', linewidth=2, color='red')
        axes4[1, 1].set_title('Evasion: Lenient vs Strict')
        axes4[1, 1].set_xlabel('Year')
        axes4[1, 1].set_ylabel('Evasion Rate')
        axes4[1, 1].legend()
        axes4[1, 1].grid(True, alpha=0.3)
        
        # Total Sugar
        axes4[1, 2].plot(lenient_data.index, lenient_data.total_sugar, 
                         label='Lenient', linewidth=2, color='green')
        axes4[1, 2].plot(strict_data.index, strict_data.total_sugar, 
                         label='Strict', linewidth=2, color='red')
        axes4[1, 2].set_title('Total Sugar: Lenient vs Strict')
        axes4[1, 2].set_xlabel('Year')
        axes4[1, 2].set_ylabel('Total Sugar')
        axes4[1, 2].legend()
        axes4[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_output/dqn_lenient_vs_strict.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. CONVERGENCE ANALYSIS
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
    fig5.suptitle('DQN Convergence Analysis', fontsize=16, fontweight='bold')
    
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp].copy()
        
        window = 50
        if len(exp_data) >= window:
            exp_data['compliance_ma'] = exp_data.compliance_rate.rolling(window=window).mean()
            axes5[0, 0].plot(exp_data.year, exp_data.compliance_ma, label=exp)
        
        if len(exp_data) >= window:
            exp_data['deaths_std'] = exp_data.deaths.rolling(window=window).std()
            axes5[0, 1].plot(exp_data.year, exp_data.deaths_std, label=exp)
        
        # Population stability
        axes5[1, 0].plot(exp_data.year, exp_data.population / exp_data.population.iloc[0], label=exp)
        
        # Convergence time
        convergence_year = exp_data[exp_data.epsilon < 0.1].year.min() if any(exp_data.epsilon < 0.1) else None
        if convergence_year is not None:
            axes5[1, 1].scatter(exp, convergence_year, s=100, alpha=0.7)
    
    axes5[0, 0].set_title(f'Compliance Convergence (MA-{window})')
    axes5[0, 0].set_xlabel('Year')
    axes5[0, 0].set_ylabel('Compliance Rate (Smoothed)')
    axes5[0, 0].legend(fontsize=8)
    axes5[0, 0].grid(True, alpha=0.3)
    
    axes5[0, 1].set_title(f'Death Rate Volatility (Std-{window})')
    axes5[0, 1].set_xlabel('Year')
    axes5[0, 1].set_ylabel('Deaths Std Dev')
    axes5[0, 1].legend(fontsize=8)
    axes5[0, 1].grid(True, alpha=0.3)
    
    axes5[1, 0].set_title('Population Retention Rate')
    axes5[1, 0].set_xlabel('Year')
    axes5[1, 0].set_ylabel('Population (Normalized)')
    axes5[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    axes5[1, 0].legend(fontsize=8)
    axes5[1, 0].grid(True, alpha=0.3)
    
    axes5[1, 1].set_title('Years to Convergence (ε < 0.1)')
    axes5[1, 1].set_ylabel('Years')
    axes5[1, 1].tick_params(axis='x', rotation=45)
    axes5[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_output/dqn_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. SUMMARY STATISTICS
    fig6, ax = plt.subplots(figsize=(14, len(summary.experiment.unique()) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    stats_data = []
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        stats_data.append([
            exp,
            f"{exp_data.population.mean():.1f}",
            f"{exp_data.gini.mean():.3f}",
            f"{exp_data.compliance_rate.mean():.3f}",
            f"{exp_data.evasion_rate.mean():.3f}",
            f"{exp_data.deaths.sum():.0f}",
            f"{exp_data.epsilon.iloc[-1]:.4f}"
        ])
    
    table = ax.table(cellText=stats_data,
                     colLabels=['Experiment', 'Avg Pop', 'Avg Gini', 'Avg Compliance', 
                               'Avg Evasion', 'Total Deaths', 'Final ε'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    plt.title('Experiment Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('analysis_output/dqn_summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Generated all visualizations in 'analysis_output/' directory:")
    print("  1. dqn_main_results.png")
    print("  2. dqn_learning_dynamics.png")
    print("  3. dqn_tax_behavior_patterns.png")
    print("  4. dqn_lenient_vs_strict.png")
    print("  5. dqn_convergence_analysis.png")
    print("  6. dqn_summary_statistics.png")
    
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("DQN Sugarscape Results Analysis")
    print("=" * 60)
    
    # Load results from CSV files
    results = load_all_results('results')
    
    if results:
        # Run analysis
        summary = analyze_dqn_results(results)
        
        if summary is not None:
            print("\n✅ Analysis complete!")
            print(f"Analyzed {len(results)} data points")
            print(f"Generated visualizations in 'analysis_output/' directory")
    else:
        print("❌ No results found to analyze")
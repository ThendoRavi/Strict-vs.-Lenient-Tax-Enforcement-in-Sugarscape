import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def load_all_results(results_dir='results', mode_filter=None):
    """
    Load all CSV files from results directory
    
    Args:
        results_dir: Directory containing CSV files
        mode_filter: 'strict', 'lenient', or None for all
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
    
    # Apply mode filter if specified
    if mode_filter:
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['mode'] == mode_filter]
        print(f"\nFiltered from {original_count} to {len(combined_df)} data points (mode={mode_filter})")
    
    print(f"\nTotal data points loaded: {len(combined_df)}")
    print(f"Experiments: {combined_df['experiment'].unique().tolist()}")
    
    # Convert to list of dictionaries
    results = combined_df.to_dict('records')
    
    return results


def analyze_dqn_results(results, output_suffix=''):
    """
    Enhanced analysis with additional visualizations
    
    Args:
        results: List of result dictionaries
        output_suffix: String to append to output filenames (e.g., '_strict_only')
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
    # MAIN SUMMARY PLOT
    # ==========================================
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    mode_label = df['mode'].iloc[0].capitalize() if 'mode' in df.columns else 'All'
    fig.suptitle(f'DQN Tax Enforcement Experiment Results - {mode_label} Mode', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Gini Coefficient
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 0].plot(exp_data.year, exp_data.gini, label=exp, linewidth=1.5)
    axes[0, 0].set_title('Wealth Inequality (Gini Coefficient)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Gini Coefficient')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Partial Payment Rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 1].plot(exp_data.year, exp_data.partial_payment_rate, label=exp, linewidth=1.5)
    axes[0, 1].set_title('Partial Payment Rate')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Partial Payment Rate')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Population
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 0].plot(exp_data.year, exp_data.population, label=exp, linewidth=1.5)
    axes[1, 0].set_title('Population Over Time')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Deaths
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 1].plot(exp_data.year, exp_data.deaths, label=exp, linewidth=1.5)
    axes[1, 1].set_title('Agent Deaths per Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Deaths')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Epsilon
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[2, 0].plot(exp_data.year, exp_data.epsilon, label=exp, linewidth=1.5)
    axes[2, 0].set_title('DQN Exploration Rate (Epsilon)')
    axes[2, 0].set_xlabel('Year')
    axes[2, 0].set_ylabel('Epsilon')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Evasion Rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[2, 1].plot(exp_data.year, exp_data.evasion_rate, label=exp, linewidth=1.5)
    axes[2, 1].set_title('Tax Evasion Rate')
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].set_ylabel('Evasion Rate')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 7: Compliance Rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[3, 0].plot(exp_data.year, exp_data.compliance_rate, label=exp, linewidth=1.5)
    axes[3, 0].set_title('Tax Compliance Rate')
    axes[3, 0].set_xlabel('Year')
    axes[3, 0].set_ylabel('Compliance Rate')
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[3, 1])
    
    plt.tight_layout()
    plt.savefig(f'analysis_output/dqn_main_results{output_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Generated: analysis_output/dqn_main_results{output_suffix}.png")
    
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("DQN Sugarscape Results Analysis")
    print("=" * 60)
    
    # Choose mode: 'strict', 'lenient', or None for all
    MODE_FILTER = 'strict'  # Change this to 'lenient' or None
    
    # Load results from CSV files with filter
    results = load_all_results('results', mode_filter=MODE_FILTER)
    
    if results:
        # Run analysis with appropriate suffix
        suffix = f'_{MODE_FILTER}' if MODE_FILTER else '_all'
        summary = analyze_dqn_results(results, output_suffix=suffix)
        
        if summary is not None:
            print("\n✅ Analysis complete!")
            print(f"Analyzed {len(results)} data points")
            print(f"Generated visualizations in 'analysis_output/' directory")
    else:
        print("❌ No results found to analyze")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynetlogo
import time
from datetime import datetime
import json
import os


class EnhancedQLearningTaxAgent:
    """Enhanced Q-Learning agent with detailed state tracking for tax compliance"""
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.8
        self.exploration_decay = 0.9995
        self.action_size = action_size
        
    def discretize_state(self, sugar_level, punished, history_len, last_action, 
                        punishment_count, evasion_success_rate, compliance_pattern):
        """
        Enhanced state discretization with detailed features:
        - sugar_level: 0-9 (10 bins)
        - punished: 0-1 (2 bins)
        - history_len: 0-4+ (5 bins)
        - last_action: -1,0,1,2 (4 bins: none, pay, partial, evade)
        - punishment_count: 0-3+ (4 bins)
        - evasion_success_rate: 0-3 (4 bins: 0%, low, medium, high)
        - compliance_pattern: 0-2 (3 bins: mixed, always comply, alternate)
        
        Total state space: 10 * 2 * 5 * 4 * 4 * 4 * 3 = 19,200 states
        """
        # Discretize each feature
        sugar_bin = min(int(sugar_level), 9)
        punished_bin = 1 if punished > 0 else 0
        history_bin = min(int(history_len / 10), 4)  # 0-9, 10-19, 20-29, 30-39, 40+
        
        # Last action: -1=none(0), 0=pay(1), 1=partial(2), 2=evade(3)
        action_bin = max(0, min(3, int(last_action) + 1))
        
        # Punishment count: 0, 1, 2, 3+
        punishment_bin = min(int(punishment_count), 3)
        
        # Evasion success rate: 0=none, 1=low(<0.3), 2=medium(0.3-0.7), 3=high(>0.7)
        if evasion_success_rate == 0:
            evasion_bin = 0
        elif evasion_success_rate < 0.3:
            evasion_bin = 1
        elif evasion_success_rate < 0.7:
            evasion_bin = 2
        else:
            evasion_bin = 3
        
        # Compliance pattern: 0=mixed, 1=always comply, 2=alternate
        pattern_bin = min(int(compliance_pattern), 2)
        
        # Encode into single state index
        state = (sugar_bin * 1920 +      # 10 sugar levels
                punished_bin * 960 +      # 2 punishment states
                history_bin * 192 +       # 5 history bins
                action_bin * 48 +         # 4 action types
                punishment_bin * 12 +     # 4 punishment counts
                evasion_bin * 3 +         # 4 evasion rates
                pattern_bin)              # 3 patterns
        
        return state
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
            
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        max_next = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_next)
        self.q_table[state][action] = new_q
        self.exploration_rate *= self.exploration_decay
    
    def get_state_info(self, state):
        """Decode state index back to features for debugging"""
        pattern_bin = state % 3
        state = state // 3
        evasion_bin = state % 4
        state = state // 4
        punishment_bin = state % 4
        state = state // 4
        action_bin = state % 4
        state = state // 4
        history_bin = state % 5
        state = state // 5
        punished_bin = state % 2
        sugar_bin = state // 2
        
        return {
            'sugar_level': sugar_bin,
            'punished': punished_bin,
            'history_bin': history_bin,
            'last_action': action_bin - 1,
            'punishment_count': punishment_bin,
            'evasion_rate': evasion_bin,
            'pattern': pattern_bin
        }


class TaxSimulation:
    """Main simulation class for tax compliance experiments"""
    def __init__(self, netlogo_path=None, gui=False):
        self.gui = gui
        try:
            if netlogo_path:
                self.netlogo = pynetlogo.NetLogoLink(gui=gui, netlogo_home=netlogo_path)
            else:
                self.netlogo = pynetlogo.NetLogoLink(gui=gui)
                
            model_file = 'Sugarscape 2 Constant Growback.nlogo'
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"NetLogo model file '{model_file}' not found")
                
            self.netlogo.load_model(model_file)
            # Enhanced state space: 19,200 states
            self.agent = EnhancedQLearningTaxAgent(state_size=19200, action_size=3)
            self.results = []
            
        except Exception as e:
            print(f"Error initializing NetLogo: {e}")
            raise
    
    def test_connection(self):
        """Test the connection and enhanced state reporting"""
        print("🔄 Starting enhanced connection test...")
        
        try:
            print("🎯 Running setup...")
            self.netlogo.command('setup')
            print("✅ Setup complete!")
            
            population = int(float(self.netlogo.report('get-population')))
            print(f"👥 Population: {population} turtles")
            
            print("📊 Getting enhanced turtle states...")
            states = self.netlogo.report('report-states')
            rewards = self.netlogo.report('report-rewards')
            
            print(f"📝 Got {len(states)} enhanced turtle states")
            print(f"💰 Got {len(rewards)} turtle rewards")
            
            print("\n🐢 First 3 turtles with enhanced features:")
            for i in range(min(3, len(states))):
                if len(states[i]) >= 8:
                    (turtle_id, sugar_level, punished, history, last_action, 
                     punishment_count, evasion_rate, pattern) = states[i]
                    reward = rewards[i]
                    print(f"  Turtle {turtle_id}:")
                    print(f"    Sugar Level={sugar_level}, Reward={reward:.1f}")
                    print(f"    Punished={punished}, Last Action={last_action}")
                    print(f"    Punishments Received={punishment_count}")
                    print(f"    Evasion Success Rate={evasion_rate:.2f}")
                    print(f"    Compliance Pattern={pattern}")
            
            print("\n🎮 Testing enhanced action system...")
            test_actions = [0] * population  # All pay
            action_string = "[" + " ".join(map(str, test_actions)) + "]"
            self.netlogo.command(f'receive-actions {action_string}')
            self.netlogo.command('go')
            
            # Check state after action
            new_states = self.netlogo.report('report-states')
            print(f"\n📈 After 1 tick - checking state changes:")
            if len(new_states) > 0 and len(new_states[0]) >= 8:
                turtle_id, sugar_level, punished, history, last_action, _, _, _ = new_states[0]
                print(f"  First turtle's last action recorded as: {last_action}")
            
            print("\n🎉 ENHANCED CONNECTION TEST SUCCESSFUL! 🎉")
            print("✅ Python receives detailed agent states")
            print("✅ Action history tracking works")
            print("✅ Punishment tracking works")
            print("✅ Evasion success tracking works")
            print("✅ Compliance pattern detection works")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_episode(self, years, audit_rate, mode, duration):
        """Run episode with enhanced state tracking"""
        try:
            self.netlogo.command(f'set-params {audit_rate} "{mode}" {duration}')
            self.netlogo.command('setup')
            
            episode_results = []
            
            for year in range(years):
                states = self.netlogo.report('report-states')
                rewards = self.netlogo.report('report-rewards')
                population = int(float(self.netlogo.report('get-population')))
                
                if len(states) == 0 or population == 0:
                    print(f"No turtles remaining at year {year}")
                    break
                
                # Store pre-states and choose actions using enhanced features
                pre_states = {}
                actions = {}
                
                for state_data in states:
                    if len(state_data) >= 8:
                        (turtle_id, sugar_level, punished, history_len, last_action,
                         punishment_count, evasion_success_rate, compliance_pattern) = state_data[:8]
                        
                        # Use enhanced state discretization
                        state = self.agent.discretize_state(
                            sugar_level, punished, history_len, last_action,
                            punishment_count, evasion_success_rate, compliance_pattern
                        )
                        action = self.agent.choose_action(state)
                        
                        pre_states[turtle_id] = state
                        actions[turtle_id] = action
                
                # Send actions
                action_list = []
                for state_data in states:
                    if len(state_data) >= 8:
                        turtle_id = state_data[0]
                        action_list.append(actions.get(turtle_id, 0))
                
                while len(action_list) < population:
                    action_list.append(0)
                action_list = action_list[:population]
                
                action_string = "[" + " ".join(map(str, action_list)) + "]"
                
                try:
                    self.netlogo.command(f'receive-actions {action_string}')
                except Exception as e:
                    print(f"  Error sending actions: {e}")
                    continue
                
                self.netlogo.command('go')
                
                # Get post-states with enhanced features
                post_states = self.netlogo.report('report-states')
                post_rewards = self.netlogo.report('report-rewards')
                
                # Update Q-values with enhanced state information
                for i, state_data in enumerate(post_states):
                    if len(state_data) >= 8:
                        (turtle_id, sugar_level, punished, history_len, last_action,
                         punishment_count, evasion_success_rate, compliance_pattern) = state_data[:8]
                        
                        post_state = self.agent.discretize_state(
                            sugar_level, punished, history_len, last_action,
                            punishment_count, evasion_success_rate, compliance_pattern
                        )
                        
                        if turtle_id in pre_states and i < len(post_rewards) and i < len(rewards):
                            reward = post_rewards[i] - rewards[i]
                            self.agent.update_q_table(
                                pre_states[turtle_id],
                                actions[turtle_id],
                                reward,
                                post_state
                            )
                
                # Collect metrics
                try:
                    gini = self.netlogo.report('calculate-gini [sugar] of turtles')
                    if gini is None:
                        gini = 0
                except:
                    gini = 0
                
                try:
                    compliance_count = self.netlogo.report('count turtles with [length compliance-history > 0 and last compliance-history = "full"]')
                    compliance = compliance_count / max(1, population)
                except:
                    compliance = 0
                
                try:
                    evasion_count = self.netlogo.report('count turtles with [strategy = "evade"]')
                    evasion = evasion_count / max(1, population)
                except:
                    evasion = 0
                
                try:
                    total_sugar = self.netlogo.report('sum [sugar] of turtles')
                    if total_sugar is None:
                        total_sugar = 0
                except:
                    total_sugar = 0
                
                # Additional metrics for enhanced tracking
                try:
                    avg_punishment_count = self.netlogo.report('mean [length punishment-history] of turtles')
                except:
                    avg_punishment_count = 0
                
                try:
                    avg_evasion_success = self.netlogo.report('mean [recent-evasion-success] of turtles')
                except:
                    avg_evasion_success = 0
                
                episode_results.append({
                    'year': year,
                    'gini': gini,
                    'compliance_rate': compliance,
                    'evasion_rate': evasion,
                    'total_sugar': total_sugar,
                    'population': population,
                    'exploration_rate': self.agent.exploration_rate,
                    'avg_punishment_count': avg_punishment_count,
                    'avg_evasion_success': avg_evasion_success
                })
                
                if year % 10 == 0:
                    print(f"Year {year}: Pop={population}, Gini={gini:.2f}, "
                          f"Comply={compliance:.2f}, AvgPunish={avg_punishment_count:.1f}")
            
            return episode_results
            
        except Exception as e:
            print(f"Error during episode: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_experiment(self, experiment_name, years=500, audit_rate=0.3, mode="strict", duration=5, repetitions=5):
        """Run complete experiment with enhanced learning"""
        experiment_results = []
        os.makedirs('results', exist_ok=True)
        
        if not hasattr(self, 'agent') or self.agent is None:
            self.agent = EnhancedQLearningTaxAgent(state_size=19200, action_size=3)
            print(f"🧠 Initialized enhanced Q-learning agent for {experiment_name}")
        else:
            print(f"🧠 Continuing with existing knowledge for {experiment_name}")
            print(f"   Current exploration rate: {self.agent.exploration_rate:.4f}")
        
        for rep in range(repetitions):
            print(f"\n🔬 Running repetition {rep+1}/{repetitions} for {experiment_name}")
            
            try:
                results = self.run_episode(years, audit_rate, mode, duration)
                
                if len(results) > 0:
                    for result in results:
                        result['repetition'] = rep
                        result['experiment'] = experiment_name
                        result['audit_rate'] = audit_rate
                        result['mode'] = mode
                        result['duration'] = duration
                    
                    experiment_results.extend(results)
                    
                    print(f"   📊 Repetition {rep+1} complete:")
                    print(f"      Exploration rate: {self.agent.exploration_rate:.6f}")
                    non_zero_q = np.count_nonzero(self.agent.q_table)
                    total_q = self.agent.q_table.size
                    print(f"      Q-values learned: {non_zero_q}/{total_q} ({100*non_zero_q/total_q:.1f}%)")
                    
                    self.export_data(experiment_name, rep)
                
            except Exception as e:
                print(f"Error in repetition {rep}: {e}")
                continue
        
        return experiment_results
    
    def export_data(self, experiment_name, repetition):
        """Export enhanced data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"results/{experiment_name}_{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            
            np.save(f"{dir_name}/qtable_rep{repetition}.npy", self.agent.q_table)
            self.netlogo.command(f'export-data "{dir_name}/agents_rep{repetition}"')
            
            non_zero_states = np.count_nonzero(np.any(self.agent.q_table != 0, axis=1))
            total_states = self.agent.q_table.shape[0]
            
            params = {
                'exploration_rate': float(self.agent.exploration_rate),
                'learning_rate': float(self.agent.learning_rate),
                'discount_factor': float(self.agent.discount_factor),
                'repetition': repetition,
                'enhanced_features': [
                    'sugar_level', 'punished_status', 'history_length',
                    'last_action', 'punishment_count', 'evasion_success_rate',
                    'compliance_pattern'
                ],
                'q_table_analytics': {
                    'states_learned': int(non_zero_states),
                    'total_states': int(total_states),
                    'learning_coverage': float(non_zero_states / total_states)
                }
            }
            
            with open(f"{dir_name}/params_rep{repetition}.json", 'w') as f:
                json.dump(params, f, indent=2)
                
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def reset_agent_for_new_experiment(self):
        """Reset agent for new experiment"""
        print("Resetting Q-learning agent for new experiment parameters")
        old_exploration = self.agent.exploration_rate if hasattr(self, 'agent') and self.agent else "N/A"
        self.agent = EnhancedQLearningTaxAgent(state_size=19200, action_size=3)
        print(f"   Previous exploration rate: {old_exploration}")
        print(f"   New exploration rate: {self.agent.exploration_rate:.4f}")
    
    def analyze_learned_strategies(self):
        """Analyze what strategies the agent has learned"""
        print("\n=== LEARNED STRATEGY ANALYSIS ===")
        
        # Find most explored states
        state_visits = np.sum(self.agent.q_table != 0, axis=1)
        most_visited_indices = np.argsort(state_visits)[-10:][::-1]
        
        print("\nTop 10 Most Explored States:")
        for i, state_idx in enumerate(most_visited_indices):
            if state_visits[state_idx] > 0:
                state_info = self.agent.get_state_info(state_idx)
                best_action = np.argmax(self.agent.q_table[state_idx])
                best_q = self.agent.q_table[state_idx][best_action]
                
                action_names = ["Pay", "Partial", "Evade"]
                print(f"\n{i+1}. State {state_idx}:")
                print(f"   Features: Sugar={state_info['sugar_level']}, "
                      f"Punished={state_info['punished']}, "
                      f"LastAction={state_info['last_action']}")
                print(f"   Punishment Count={state_info['punishment_count']}, "
                      f"Pattern={state_info['pattern']}")
                print(f"   Best Action: {action_names[best_action]} (Q={best_q:.2f})")
                print(f"   All Q-values: Pay={self.agent.q_table[state_idx][0]:.2f}, "
                      f"Partial={self.agent.q_table[state_idx][1]:.2f}, "
                      f"Evade={self.agent.q_table[state_idx][2]:.2f}")
    
    def close(self):
        """Clean up NetLogo connection"""
        try:
            self.netlogo.kill_workspace()
        except:
            pass


def analyze_results(results):
    """Analyze and visualize experiment results with enhanced metrics"""
    if len(results) == 0:
        print("No results generated")
        return None
        
    df = pd.DataFrame(results)
    
    summary = df.groupby(['experiment', 'year']).agg({
        'gini': 'mean',
        'compliance_rate': 'mean',
        'evasion_rate': 'mean',
        'total_sugar': 'mean',
        'population': 'mean',
        'avg_punishment_count': 'mean',
        'avg_evasion_success': 'mean'
    }).reset_index()
    
    # Create enhanced visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Gini coefficient
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 0].plot(exp_data.year, exp_data.gini, label=exp)
    axes[0, 0].set_title('Wealth Inequality (Gini Coefficient)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Gini Coefficient')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Compliance rate
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 1].plot(exp_data.year, exp_data.compliance_rate, label=exp)
    axes[0, 1].set_title('Tax Compliance Rate')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Compliance Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Population
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 0].plot(exp_data.year, exp_data.population, label=exp)
    axes[1, 0].set_title('Population Over Time')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total sugar
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 1].plot(exp_data.year, exp_data.total_sugar, label=exp)
    axes[1, 1].set_title('Total Sugar in System')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Total Sugar')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Average punishment count (NEW)
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[2, 0].plot(exp_data.year, exp_data.avg_punishment_count, label=exp)
    axes[2, 0].set_title('Average Punishment Count per Agent')
    axes[2, 0].set_xlabel('Year')
    axes[2, 0].set_ylabel('Avg Punishments Received')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Average evasion success (NEW)
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[2, 1].plot(exp_data.year, exp_data.avg_evasion_success, label=exp)
    axes[2, 1].set_title('Average Evasion Success Rate')
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].set_ylabel('Avg Successful Evasions')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/enhanced_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary


def main():
    """Main execution function"""
    netlogo_path = None
    
    TEST_MODE = True
    USE_GUI = True
    
    try:
        print("="*60)
        print("ENHANCED TAX COMPLIANCE SIMULATION WITH Q-LEARNING")
        print("="*60)
        print("\nEnhanced Features:")
        print("- Action history tracking")
        print("- Punishment consequence learning")
        print("- Evasion success tracking")
        print("- Compliance pattern detection")
        
        print("\nInitializing simulation...")
        sim = TaxSimulation(netlogo_path, gui=USE_GUI)
        
        print("\nTesting Python-NetLogo connection...")
        if not sim.test_connection():
            print("Connection test failed. Exiting...")
            sim.close()
            return
        
        print("\n" + "="*60)
        
        if TEST_MODE:
            print("\nRUNNING TEST EXPERIMENT")
            print("-"*40)
            
            experiments = [
                {
                    'name': 'enhanced_test',
                    'audit_rate': 0.5,
                    'mode': 'strict',
                    'duration': 10,
                    'years': 1000,
                    'repetitions': 2
                }
            ]
        else:
            print("\nRUNNING FULL EXPERIMENTS")
            print("-"*40)
            
            experiments = [
                {
                    'name': 'low_audit_enhanced',
                    'audit_rate': 0.3,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 1000,
                    'repetitions': 3
                },
                {
                    'name': 'high_audit_enhanced',
                    'audit_rate': 0.8,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 1000,
                    'repetitions': 3
                }
            ]
        
        all_results = []
        
        for i, exp in enumerate(experiments):
            print(f"\nExperiment {i+1}/{len(experiments)}: {exp['name']}")
            print(f"   Parameters: audit={exp['audit_rate']}, mode={exp['mode']}, duration={exp['duration']}")
            
            if i > 0:
                sim.reset_agent_for_new_experiment()
            
            results = sim.run_experiment(
                experiment_name=exp['name'],
                years=exp['years'],
                audit_rate=exp['audit_rate'],
                mode=exp['mode'],
                duration=exp['duration'],
                repetitions=exp['repetitions']
            )
            all_results.extend(results)
            
            print(f"   Experiment '{exp['name']}' complete!")
        
        print("\n" + "="*60)
        print("ANALYZING RESULTS")
        print("-"*40)
        
        if len(all_results) > 0:
            summary = analyze_results(all_results)
            if summary is not None:
                summary.to_csv('results/enhanced_experiment_summary.csv', index=False)
                print("Results saved to results/enhanced_experiment_summary.csv")
                print("Plots saved to results/enhanced_summary_plot.png")
                
                print("\nSummary Statistics:")
                print("-"*40)
                final_year = summary.groupby('experiment').tail(1)
                for _, row in final_year.iterrows():
                    print(f"\n{row['experiment']}:")
                    print(f"  Final Gini: {row['gini']:.3f}")
                    print(f"  Final Compliance: {row['compliance_rate']:.3f}")
                    print(f"  Final Population: {row['population']:.0f}")
                    print(f"  Avg Punishments: {row['avg_punishment_count']:.2f}")
                    print(f"  Avg Evasion Success: {row['avg_evasion_success']:.2f}")
            
            # Analyze learned strategies
            sim.analyze_learned_strategies()
        else:
            print("No results generated")
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE!")
        print("="*60)
        
        sim.close()
        
    except FileNotFoundError as e:
        print(f"\nFILE ERROR: {e}")
        print("\nPlease check:")
        print("1. 'Sugarscape 2 Constant Growback.nlogo' is in current directory")
        print("2. All file paths are correct")
        
    except Exception as e:
        print(f"\nEXECUTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure NetLogo is installed")
        print("2. Install required packages: pip install pynetlogo numpy pandas matplotlib")
        print("3. Close any open NetLogo instances")


if __name__ == "__main__":
    main()
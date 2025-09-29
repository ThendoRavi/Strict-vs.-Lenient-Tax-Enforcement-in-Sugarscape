import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynetlogo
import time
from datetime import datetime
import json
import os


class QLearningTaxAgent:
    """Q-Learning agent for tax compliance decisions"""
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.8
        self.exploration_decay = 0.9995
        self.action_size = action_size
        
    def discretize_state(self, sugar_level, punished, history_len):
        # Discretize state components
        sugar_bin = min(int(sugar_level), 9)  # Ensure integer, 0-9
        punished_bin = 0 if punished == 0 else 1  # 0 or 1
        history_bin = min(int(history_len), 2)  # 0, 1, or 2+
        
        # Combine into a single state index
        return sugar_bin * 20 + punished_bin * 10 + history_bin
        
    def choose_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
            
    def update_q_table(self, state, action, reward, next_state):
        max_next = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_next)
        self.q_table[state][action] = new_q
        self.exploration_rate *= self.exploration_decay


class TaxSimulation:
    """Main simulation class for tax compliance experiments"""
    def __init__(self, netlogo_path=None, gui=False):
        # Initialize NetLogo connection
        self.gui = gui
        try:
            if netlogo_path:
                self.netlogo = pynetlogo.NetLogoLink(gui=gui, netlogo_home=netlogo_path)
            else:
                self.netlogo = pynetlogo.NetLogoLink(gui=gui)
                
            # Check if model file exists
            model_file = 'Sugarscape 2 Constant Growback.nlogo'
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"NetLogo model file '{model_file}' not found in current directory")
                
            self.netlogo.load_model(model_file)
            self.agent = QLearningTaxAgent(state_size=200, action_size=3)
            self.results = []
            
        except Exception as e:
            print(f"Error initializing NetLogo: {e}")
            raise
    
    def test_connection(self):
        """Test the connection between Python and NetLogo"""
        print("🔄 Starting connection test...")
        
        try:
            # Step 1: Setup the simulation
            print("🎯 Running setup...")
            self.netlogo.command('setup')
            print("✅ Setup complete!")
            
            # Step 2: Check if we have turtles
            population = self.netlogo.report('get-population')
            population_int = int(float(population))  # Convert JDouble to Python int via float
            print(f"👥 Population: {population_int} turtles")
            
            # Step 3: Get some basic info
            print("📊 Getting turtle states...")
            states = self.netlogo.report('report-states')
            rewards = self.netlogo.report('report-rewards')
            
            print(f"📝 Got {len(states)} turtle states")
            print(f"💰 Got {len(rewards)} turtle rewards")
            
            # Step 4: Show first few turtles
            print("\n🐢 First 3 turtles:")
            for i in range(min(3, len(states))):
                turtle_id, sugar_level, punished, history = states[i]
                reward = rewards[i]
                print(f"  Turtle {turtle_id}: Sugar Level={sugar_level}, Reward={reward:.1f}, Punished={punished}")
            
            # Step 5: Test sending simple actions
            print("\n🎮 Testing action sending...")
            # Send action 0 (pay) to all turtles
            simple_actions = [2] * population_int  # All turtles will "pay"
            action_string = "[" + " ".join(map(str, simple_actions)) + "]"
            
            print(f"📤 Sending actions: {action_string[:50]}...")
            self.netlogo.command(f'receive-actions {action_string}')
            print("✅ Actions sent successfully!")
            
            # Step 6: Run simulation for 10 ticks as a test
            print("⏭️ Running simulation for 10 test ticks...")
            
            for tick in range(10):
                # Send actions for this tick
                current_population = self.netlogo.report('get-population')
                current_population_int = int(float(current_population))
                
                if current_population_int == 0:
                    print(f"❌ Population died out at tick {tick}")
                    break
                
                # Send actions to all current turtles
                tick_actions = [2] * current_population_int  # All turtles will "pay"
                action_string = "[" + " ".join(map(str, tick_actions)) + "]"
                self.netlogo.command(f'receive-actions {action_string}')
                
                # Run one step
                self.netlogo.command('go')
                
                # Print progress
                if (tick + 1) % 5 == 0:
                    temp_population = self.netlogo.report('get-population')
                    temp_population_int = int(float(temp_population))
                    if temp_population_int > 0:
                        temp_rewards = self.netlogo.report('report-rewards')
                        avg_sugar = sum(temp_rewards) / len(temp_rewards)
                        print(f"  Tick {tick + 1:3d}: Population={temp_population_int}, Avg Sugar={avg_sugar:.1f}")
            
            print("\n🎉 CONNECTION TEST SUCCESSFUL! 🎉")
            print("✅ Python can talk to NetLogo")
            print("✅ NetLogo can receive actions")
            print("✅ NetLogo can send back data")
            print("\nReady to run full experiments!")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
            
    def run_episode(self, years=1, audit_rate=0.3, mode="strict", duration=5):
        """Run a single episode of the simulation"""
        try:
            # Set parameters in NetLogo
            self.netlogo.command(f'set-params {audit_rate} "{mode}" {duration}')
            self.netlogo.command('setup')
            
            episode_results = []
            
            for year in range(years):
                # Get current states
                states = self.netlogo.report('report-states')
                rewards = self.netlogo.report('report-rewards')
                population = self.netlogo.report('get-population')
                
                # Convert population to int
                population = int(float(population))
                
                # Debug info
                print(f"  Debug - Year {year}: States count: {len(states)}, Population: {population}")
                
                # Handle empty states
                if len(states) == 0 or population == 0:
                    print(f"No turtles remaining at year {year}, ending episode")
                    break
                
                # Validate data consistency
                if len(states) != population:
                    print(f"  Warning: States count ({len(states)}) != Population ({population})")
                    # Use the smaller value to be safe
                    population = min(len(states), population)
                
                # Store pre-states and choose actions
                pre_states = {}
                actions = {}
                
                for state_data in states:
                    if len(state_data) >= 4:  # Ensure proper state format
                        turtle_id, sugar_level, punished, history_len = state_data[:4]
                        state = self.agent.discretize_state(sugar_level, punished, history_len)
                        action = self.agent.choose_action(state)
                        
                        pre_states[turtle_id] = state
                        actions[turtle_id] = action
                
                # Send actions to NetLogo - create action list matching turtle order
                action_list = []
                turtle_ids = []
                
                # Build ordered list of actions based on states order
                for state_data in states:
                    if len(state_data) >= 4:
                        turtle_id = state_data[0]
                        turtle_ids.append(turtle_id)
                        action_list.append(actions.get(turtle_id, 0))  # Default to action 0 if not found
                
                # Pad or trim to match population size
                while len(action_list) < population:
                    action_list.append(0)
                action_list = action_list[:population]
                
                # Convert to NetLogo list format
                action_string = "[" + " ".join(map(str, action_list)) + "]"
                
                # Debug: print action info
                print(f"  Debug - Sending {len(action_list)} actions for {population} turtles")
                print(f"  Debug - Action string preview: {action_string[:100]}...")
                
                # Send actions with error handling
                try:
                    self.netlogo.command(f'receive-actions {action_string}')
                except Exception as e:
                    print(f"  Error sending actions: {e}")
                    print(f"  Skipping year {year}")
                    continue
                
                # Run one tick (which processes the actions)
                self.netlogo.command('go')
                
                # Get post-states and rewards
                post_states = self.netlogo.report('report-states')
                post_rewards = self.netlogo.report('report-rewards')
                
                # Update Q-values
                for i, state_data in enumerate(post_states):
                    if len(state_data) >= 4:
                        turtle_id, sugar_level, punished, history_len = state_data[:4]
                        post_state = self.agent.discretize_state(sugar_level, punished, history_len)
                        
                        if turtle_id in pre_states and i < len(post_rewards) and i < len(rewards):
                            reward = post_rewards[i] - rewards[i]  # Change in sugar
                            self.agent.update_q_table(
                                pre_states[turtle_id],
                                actions[turtle_id],
                                reward,
                                post_state
                            )
                
                # Collect metrics - with error handling
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
                
                episode_results.append({
                    'year': year,
                    'gini': gini,
                    'compliance_rate': compliance,
                    'evasion_rate': evasion,
                    'total_sugar': total_sugar,
                    'population': population,
                    'exploration_rate': self.agent.exploration_rate
                })
                
                if year % 10 == 0:  # Print every 10 years
                    print(f"Year {year}: Population={population}, Gini={gini:.2f}, Compliance={compliance:.2f}")
            
            return episode_results
            
        except Exception as e:
            print(f"Error during episode: {e}")
            return []
    
    def run_experiment(self, experiment_name, years=500, audit_rate=0.3, mode="strict", duration=5, repetitions=5):
        """Run a complete experiment with multiple repetitions"""
        experiment_results = []
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        for rep in range(repetitions):
            print(f"\n🔬 Running repetition {rep+1}/{repetitions} for {experiment_name}")
            
            try:
                # Reset Q-learning for each repetition
                self.agent = QLearningTaxAgent(state_size=200, action_size=3)
                
                # Run the experiment
                results = self.run_episode(years, audit_rate, mode, duration)
                
                if len(results) > 0:  # Only add if we got valid results
                    # Add repetition info to results
                    for result in results:
                        result['repetition'] = rep
                        result['experiment'] = experiment_name
                        result['audit_rate'] = audit_rate
                        result['mode'] = mode
                        result['duration'] = duration
                    
                    experiment_results.extend(results)
                    
                    # Export data after each repetition
                    self.export_data(experiment_name, rep)
                
            except Exception as e:
                print(f"Error in repetition {rep}: {e}")
                continue
        
        return experiment_results
    
    def export_data(self, experiment_name, repetition):
        """Export simulation data and Q-table"""
        try:
            # Create directory for results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"results/{experiment_name}_{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            
            # Export Q-table
            np.save(f"{dir_name}/qtable_rep{repetition}.npy", self.agent.q_table)
            
            # Export agent data from NetLogo
            self.netlogo.command(f'export-data "{dir_name}/agents_rep{repetition}"')
            
            # Export simulation parameters
            params = {
                'exploration_rate': float(self.agent.exploration_rate),
                'learning_rate': float(self.agent.learning_rate),
                'discount_factor': float(self.agent.discount_factor)
            }
            
            with open(f"{dir_name}/params_rep{repetition}.json", 'w') as f:
                json.dump(params, f)
                
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    def close(self):
        """Clean up NetLogo connection"""
        try:
            self.netlogo.kill_workspace()
        except:
            pass


def analyze_results(results):
    """Analyze and visualize experiment results"""
    if len(results) == 0:
        print("❌ No results generated")
        return None
        
    df = pd.DataFrame(results)
    
    # Summary statistics by experiment
    summary = df.groupby(['experiment', 'year']).agg({
        'gini': 'mean',
        'compliance_rate': 'mean',
        'evasion_rate': 'mean',
        'total_sugar': 'mean',
        'population': 'mean'
    }).reset_index()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gini coefficient over time
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 0].plot(exp_data.year, exp_data.gini, label=exp)
    axes[0, 0].set_title('Wealth Inequality (Gini Coefficient)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Gini Coefficient')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Compliance rate over time
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[0, 1].plot(exp_data.year, exp_data.compliance_rate, label=exp)
    axes[0, 1].set_title('Tax Compliance Rate')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Compliance Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Population over time
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 0].plot(exp_data.year, exp_data.population, label=exp)
    axes[1, 0].set_title('Population Over Time')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Population')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total sugar over time
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        axes[1, 1].plot(exp_data.year, exp_data.total_sugar, label=exp)
    axes[1, 1].set_title('Total Sugar in System')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Total Sugar')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary


def main():
    """Main execution function with test mode and full experiment mode"""
    # Set your NetLogo path if needed
    # netlogo_path = "C:/Program Files/NetLogo 6.3.0"
    netlogo_path = None  # Auto-detect if None
    
    # Configuration
    TEST_MODE = True  # Set to False to run full experiments
    USE_GUI = False   # Set to True to see NetLogo GUI
    
    try:
        print("="*60)
        print("TAX COMPLIANCE SIMULATION WITH Q-LEARNING")
        print("="*60)
        
        # Initialize simulation
        print("\n📦 Initializing simulation...")
        sim = TaxSimulation(netlogo_path, gui=USE_GUI)
        
        # Run connection test first
        print("\n🔗 Testing Python-NetLogo connection...")
        if not sim.test_connection():
            print("❌ Connection test failed. Exiting...")
            sim.close()
            return
        
        print("\n" + "="*60)
        
        if TEST_MODE:
            # Run a quick test experiment
            print("\n🧪 RUNNING TEST EXPERIMENT")
            print("-"*40)
            
            experiments = [
                {
                    'name': 'test_run',
                    'audit_rate': 0.3,
                    'mode': 'strict',
                    'duration': 5,
                    'years': 20,  # Quick test
                    'repetitions': 1  # Single run
                }
            ]
        else:
            # Run full experiments
            print("\n🚀 RUNNING FULL EXPERIMENTS")
            print("-"*40)
            
            experiments = [
                {
                    'name': 'low_audit',
                    'audit_rate': 0.1,
                    'mode': 'strict',
                    'duration': 5,
                    'years': 100,
                    'repetitions': 3
                },
                {
                    'name': 'medium_audit',
                    'audit_rate': 0.3,
                    'mode': 'strict',
                    'duration': 5,
                    'years': 100,
                    'repetitions': 3
                },
                {
                    'name': 'high_audit',
                    'audit_rate': 0.5,
                    'mode': 'strict',
                    'duration': 5,
                    'years': 100,
                    'repetitions': 3
                },
                {
                    'name': 'lenient_mode',
                    'audit_rate': 0.3,
                    'mode': 'lenient',
                    'duration': 3,
                    'years': 100,
                    'repetitions': 3
                }
            ]
        
        all_results = []
        
        # Run all experiments
        for i, exp in enumerate(experiments):
            print(f"\n📊 Experiment {i+1}/{len(experiments)}: {exp['name']}")
            print(f"   Parameters: audit_rate={exp['audit_rate']}, mode={exp['mode']}, duration={exp['duration']}")
            print(f"   Running for {exp['years']} years, {exp['repetitions']} repetition(s)")
            
            results = sim.run_experiment(
                experiment_name=exp['name'],
                years=exp['years'],
                audit_rate=exp['audit_rate'],
                mode=exp['mode'],
                duration=exp['duration'],
                repetitions=exp['repetitions']
            )
            all_results.extend(results)
            
            print(f"   ✅ Experiment '{exp['name']}' complete!")
        
        # Analyze and visualize results
        print("\n" + "="*60)
        print("📈 ANALYZING RESULTS")
        print("-"*40)
        
        if len(all_results) > 0:
            summary = analyze_results(all_results)
            if summary is not None:
                # Save results
                summary.to_csv('results/experiment_summary.csv', index=False)
                print("✅ Results saved to results/experiment_summary.csv")
                print("✅ Plots saved to results/summary_plot.png")
                
                # Print summary statistics
                print("\n📊 Summary Statistics:")
                print("-"*40)
                final_year = summary.groupby('experiment').tail(1)
                for _, row in final_year.iterrows():
                    print(f"\n{row['experiment']}:")
                    print(f"  Final Gini: {row['gini']:.3f}")
                    print(f"  Final Compliance Rate: {row['compliance_rate']:.3f}")
                    print(f"  Final Population: {row['population']:.0f}")
                    print(f"  Final Total Sugar: {row['total_sugar']:.0f}")
        else:
            print("❌ No results generated")
        
        print("\n" + "="*60)
        print("🎉 SIMULATION COMPLETE!")
        print("="*60)
        
        # Clean up
        sim.close()
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
        print("\n🔍 Please check:")
        print("1. The 'Sugarscape 2 Constant Growback.nlogo' file is in the current directory")
        print("2. All file paths are correct")
        
    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Make sure NetLogo is installed")
        print("2. Make sure pynetlogo is installed: pip install pynetlogo")
        print("3. Check that all required Python packages are installed:")
        print("   - numpy, pandas, matplotlib, pynetlogo")
        print("4. Try closing NetLogo if it's already open")
        print("5. Check the NetLogo model has the required procedures:")
        print("   - setup, go, get-population, report-states, report-rewards")
        print("   - receive-actions, set-params, export-data, calculate-gini")


if __name__ == "__main__":
    main()
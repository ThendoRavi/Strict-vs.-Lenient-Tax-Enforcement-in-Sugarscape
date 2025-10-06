import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynetlogo
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
from datetime import datetime
import json
import os
import time


class DQNAgent:
    """Deep Q-Network agent for tax compliance learning"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration parameter
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # discount factor
        self.batch_size = 32
        self.update_target_every = 100
        self.step_count = 0
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self):
        """Build the neural network for Q-value approximation"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = tf.expand_dims(state, 0)
        q_values = self.q_network(state_tensor, training=False)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Predict Q-values for starting states
        current_q_values = self.q_network(states, training=False).numpy()
        
        # Predict Q-values for next states using target network (Double DQN)
        next_q_values = self.target_network(next_states, training=False).numpy()
        
        # Update Q-values with Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the network
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()
    
    def save(self, filepath):
        """Save the model weights"""
        self.q_network.save_weights(filepath)
    
    def load(self, filepath):
        """Load the model weights"""
        self.q_network.load_weights(filepath)
        self.update_target_network()


class MultiAgentDQNEnvironment:
    """Environment manager for multi-agent DQN learning with comprehensive action space"""
    
    def __init__(self, n_agents=200):
        self.n_agents = n_agents
        self.state_size = 10  # Normalized state features
        
        # Action space definitions
        self.movement_actions = 4  # UP, DOWN, LEFT, RIGHT
        self.consumption_actions = 2  # CONSUME, HARVEST
        self.tax_actions = 3  # PAY_FULL, PAY_PARTIAL, EVADE
        
        # Total action space: Movement + Consumption + Tax (when applicable)
        self.total_action_size = self.movement_actions + self.consumption_actions + self.tax_actions
        
        # Create comprehensive DQN agent
        self.agent = DQNAgent(self.state_size, self.total_action_size)
        
        # Track agent states and histories
        self.agent_states = {}
        self.agent_histories = {}
        self.death_penalties = -100  # Large negative reward for death
        self.current_tick = 0
        self.audit_frequency = 50  # Audit every 50 ticks
        
        # Action mapping
        self.action_mapping = {
            # Movement actions (0-3)
            0: 'UP',
            1: 'DOWN', 
            2: 'LEFT',
            3: 'RIGHT',
            # Consumption actions (4-5)
            4: 'CONSUME',
            5: 'HARVEST',
            # Tax actions (6-8) - only during audit periods
            6: 'PAY_FULL',
            7: 'PAY_PARTIAL',
            8: 'EVADE'
        }
        
    def normalize_state(self, sugar_level, x_pos, y_pos, punished, punishment_history, metabolism=2):
        """
        Normalize state features for neural network input
        
        Enhanced state components:
        - Sugar storage (normalized 0-1) - total accumulated sugar
        - Health status (normalized 0-1) - based on sugar relative to metabolism needs
        - X position (normalized 0-1)
        - Y position (normalized 0-1) 
        - Punishment status (0 or 1)
        - Last 3 punishment decisions encoded (normalized 0-1 each)
        - Current tick modulo audit frequency (for timing awareness)
        - Distance to nearest high-sugar area (placeholder)
        """
        # Sugar storage - raw amount normalized
        max_expected_sugar = 100.0
        sugar_storage = min(sugar_level / max_expected_sugar, 1.0)
        
        # Health status - sugar relative to survival needs
        # Health is critical when sugar < metabolism * 5 turns
        survival_threshold = metabolism * 5
        health_status = min(sugar_level / survival_threshold, 1.0) if survival_threshold > 0 else 1.0
        
        # Normalize grid positions (50x50 grid)
        norm_x = x_pos / 49.0  # 0-49 grid
        norm_y = y_pos / 49.0
        
        # Punishment status
        punishment_flag = float(punished)
        
        # Encode last 3 punishment history decisions
        history_encoding = np.zeros(3)
        for i, hist_idx in enumerate([-1, -2, -3]):
            if len(punishment_history) >= abs(hist_idx):
                action = punishment_history[hist_idx]
                history_encoding[i] = action / 2.0 if action >= 0 else 0.0  # Normalize 0,1,2 to 0,0.5,1
        
        # Audit timing awareness - where we are in audit cycle
        audit_phase = (self.current_tick % self.audit_frequency) / self.audit_frequency
        
        # Resource scarcity indicator (placeholder - could be enhanced)
        resource_scarcity = 0.5  # Neutral value, could be calculated from nearby patches
        
        state = np.array([
            sugar_storage,      # 0: Raw sugar amount
            health_status,      # 1: Health relative to survival needs  
            norm_x,            # 2: X position
            norm_y,            # 3: Y position
            punishment_flag,   # 4: Currently punished?
            history_encoding[0], # 5: Last tax decision
            history_encoding[1], # 6: 2nd last tax decision
            history_encoding[2], # 7: 3rd last tax decision
            audit_phase,       # 8: Current phase in audit cycle
            resource_scarcity  # 9: Resource availability indicator
        ])
        
        return state
    
    def process_netlogo_states(self, raw_states):
        """Process raw NetLogo states into normalized neural network inputs"""
        processed_states = {}
        
        for state_data in raw_states:
            if len(state_data) >= 8:
                turtle_id = int(state_data[0])
                sugar_level = state_data[1]
                punished = state_data[2]
                history_len = state_data[3]
                last_action = state_data[4]
                punishment_count = state_data[5]
                
                # Extract positions if available (new format with 10 elements)
                if len(state_data) >= 10:
                    evasion_success_rate = state_data[6]
                    compliance_pattern = state_data[7]
                    x_pos = state_data[8]
                    y_pos = state_data[9]
                else:
                    # Fallback for older format without positions
                    x_pos = np.random.randint(0, 50)
                    y_pos = np.random.randint(0, 50)
                
                # Build punishment history from last actions
                if turtle_id not in self.agent_histories:
                    self.agent_histories[turtle_id] = []
                
                if last_action >= 0:
                    self.agent_histories[turtle_id].append(last_action)
                    if len(self.agent_histories[turtle_id]) > 3:
                        self.agent_histories[turtle_id] = self.agent_histories[turtle_id][-3:]
                
                # Normalize state
                normalized_state = self.normalize_state(
                    sugar_level, x_pos, y_pos, punished, 
                    self.agent_histories.get(turtle_id, [])
                )
                
                processed_states[turtle_id] = {
                    'state': normalized_state,
                    'sugar_level': sugar_level,
                    'punished': punished,
                    'raw_data': state_data
                }
                
        return processed_states
    
    def get_legal_actions(self, is_audit_period=False, is_punished=False):
        """Get mask for legal actions based on current game state"""
        legal_mask = np.zeros(self.total_action_size, dtype=bool)
        
        # Movement actions (0-3): Always legal unless punished
        if not is_punished:
            legal_mask[0:4] = True  # UP, DOWN, LEFT, RIGHT
        
        # Consumption/Harvest actions (4-5): Always legal
        legal_mask[4:6] = True  # CONSUME, HARVEST
        
        # Tax actions (6-8): Only legal during audit periods
        if is_audit_period:
            legal_mask[6:9] = True  # PAY_FULL, PAY_PARTIAL, EVADE
        
        return legal_mask

    def choose_actions(self, processed_states, is_audit_period=False):
        """Choose actions for all agents with legal action masking - OPTIMIZED"""
        actions = {}
        
        if not processed_states:
            return actions
        
        # Batch processing for efficiency
        turtle_ids = list(processed_states.keys())
        states_batch = np.array([processed_states[tid]['state'] for tid in turtle_ids])
        punished_batch = [bool(processed_states[tid]['punished']) for tid in turtle_ids]
        
        # Get Q-values for all states in one batch prediction (much faster!)
        q_values_batch = self.agent.q_network.predict(states_batch, verbose=0)
        
        for i, turtle_id in enumerate(turtle_ids):
            is_punished = punished_batch[i]
            q_values = q_values_batch[i]
            
            # Get legal actions for this agent
            legal_mask = self.get_legal_actions(is_audit_period, is_punished)
            
            # Mask illegal actions with very negative values
            masked_q_values = np.where(legal_mask, q_values, -np.inf)
            
            # Choose action using epsilon-greedy with masked Q-values
            if np.random.random() <= self.agent.epsilon:
                # Random action from legal actions only
                legal_actions = np.where(legal_mask)[0]
                action = np.random.choice(legal_actions)
            else:
                # Best legal action
                action = np.argmax(masked_q_values)
            
            actions[turtle_id] = action
            
        return actions
    
    def translate_actions_to_netlogo(self, actions, is_audit_period=False):
        """
        Translate DQN actions to NetLogo-compatible format
        Returns movement commands and tax decisions separately
        """
        movement_commands = []
        tax_decisions = []
        
        for turtle_id, action in actions.items():
            # Handle movement actions (0-3)
            if action in [0, 1, 2, 3]:
                direction = self.action_mapping[action]
                movement_commands.append((turtle_id, direction))
            
            # Handle tax actions (6-8) - only during audit periods
            elif action in [6, 7, 8] and is_audit_period:
                if action == 6:  # PAY_FULL
                    tax_decisions.append((turtle_id, 0))  # NetLogo action 0
                elif action == 7:  # PAY_PARTIAL  
                    tax_decisions.append((turtle_id, 1))  # NetLogo action 1
                elif action == 8:  # EVADE
                    tax_decisions.append((turtle_id, 2))  # NetLogo action 2
            
            # Consumption/Harvest actions (4-5) are handled automatically by NetLogo
        
        return movement_commands, tax_decisions
    
    def calculate_rewards(self, pre_states, post_states, actions, died_agents):
        """
        Calculate rewards for all agents
        
        New reward structure:
        - 0 for most actions (movement, consumption, harvesting)
        - Large negative reward (-100) for death
        - Moderate negative reward (-20) for complete tax evasion when caught
        - Small negative reward (-10) for partial payment when caught
        - Zero reward for full tax payment (compliance)
        """
        rewards = {}
        
        for turtle_id in pre_states:
            if turtle_id in died_agents:
                # Agent died - large negative reward to discourage death
                rewards[turtle_id] = self.death_penalties
            elif turtle_id in post_states:
                # Agent survived - calculate reward based on actions and consequences
                pre_punished = pre_states[turtle_id]['punished']
                post_punished = post_states[turtle_id]['punished']
                action = actions.get(turtle_id, 0)
                
                reward = 0.0  # Default: zero reward for most actions
                
                # Punishment-based rewards (only when punishment status changes)
                if not pre_punished and post_punished:
                    # Agent got punished this turn - check what tax action they took
                    if action == 8:  # EVADE action (index 8)
                        reward = -20.0  # Severe punishment for complete evasion
                    elif action == 7:  # PAY_PARTIAL action (index 7)
                        reward = -10.0  # Less severe punishment for partial payment
                    # PAY_FULL (index 6) gets 0 reward (no additional penalty)
                
                rewards[turtle_id] = reward
            else:
                # Agent disappeared (shouldn't happen)
                rewards[turtle_id] = self.death_penalties
                
        return rewards
    
    def train_step(self, pre_states, actions, rewards, post_states):
        """Train the DQN agents with collected experiences"""
        for turtle_id in pre_states:
            if turtle_id in actions:
                pre_state = pre_states[turtle_id]['state']
                action = actions[turtle_id]
                reward = rewards.get(turtle_id, 0)
                done = turtle_id not in post_states or reward == self.death_penalties
                
                if done:
                    # Agent died or episode ended
                    next_state = np.zeros(self.state_size)
                else:
                    next_state = post_states[turtle_id]['state']
                
                # Store experience and train
                self.agent.remember(pre_state, action, reward, next_state, done)
        
        # Perform experience replay
        self.agent.replay()


class DQNTaxSimulation:
    """Main simulation class with DQN agents"""
    
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
            self.env = MultiAgentDQNEnvironment()
            self.results = []
            
        except Exception as e:
            print(f"Error initializing NetLogo: {e}")
            raise
    
    def run_episode(self, years, audit_rate, mode, duration):
        """Run a single episode with enhanced multi-action DQN agents"""
        try:
            self.netlogo.command(f'set-params {audit_rate} "{mode}" {duration}')
            self.netlogo.command('setup')
            
            episode_results = []
            consecutive_empty_years = 0
            
            for year in range(years):
                # Update environment tick counter
                self.env.current_tick = year
                
                # Get current states
                raw_states = self.netlogo.report('report-states')
                population = int(float(self.netlogo.report('get-population')))
                
                if len(raw_states) == 0 or population == 0:
                    print(f"No turtles remaining at year {year}")
                    consecutive_empty_years += 1
                    if consecutive_empty_years > 5:
                        break
                    continue
                
                consecutive_empty_years = 0
                
                # Process states for neural network
                pre_states = self.env.process_netlogo_states(raw_states)
                
                # Determine if it's an audit period (every 50 ticks)
                is_audit_period = (year % self.env.audit_frequency == 0)
                
                # Choose actions using enhanced DQN with legal action masking
                if year < 3:  # Debug info for first few years
                    print(f"  Choosing actions for {len(pre_states)} agents...")
                actions_start = time.time()
                actions = self.env.choose_actions(pre_states, is_audit_period)
                actions_time = time.time() - actions_start
                if year < 3:
                    print(f"  Action selection took {actions_time:.2f}s")
                
                # Translate actions to NetLogo commands
                movement_commands, tax_decisions = self.env.translate_actions_to_netlogo(actions, is_audit_period)
                
                # Send movement commands to NetLogo (if any)
                if movement_commands:
                    # Format: [[turtle-id direction] [turtle-id direction] ...]
                    movement_string = "["
                    for turtle_id, direction in movement_commands:
                        movement_string += f"[{turtle_id} \"{direction}\"] "
                    movement_string += "]"
                    self.netlogo.command(f'receive-movement-commands {movement_string}')
                
                # Send tax decisions to NetLogo (only during audit periods)
                if is_audit_period and tax_decisions:
                    # Convert tax decisions to standard action list format
                    tax_action_list = [0] * population  # Default to PAY_FULL
                    turtle_list = [int(state[0]) for state in raw_states]
                    
                    for turtle_id, tax_action in tax_decisions:
                        if turtle_id in turtle_list:
                            idx = turtle_list.index(turtle_id)
                            if idx < len(tax_action_list):
                                tax_action_list[idx] = tax_action
                    
                    action_string = "[" + " ".join(map(str, tax_action_list)) + "]"
                    self.netlogo.command(f'receive-actions {action_string}')
                
                # Execute NetLogo simulation step
                self.netlogo.command('go')
                
                # Get post-states
                post_raw_states = self.netlogo.report('report-states')
                rewards_after = self.netlogo.report('report-rewards')
                post_population = int(float(self.netlogo.report('get-population')))
                
                # Process post-states
                post_states = self.env.process_netlogo_states(post_raw_states)
                
                # Identify agents that died
                died_agents = set(pre_states.keys()) - set(post_states.keys())
                
                # Calculate rewards
                rewards = self.env.calculate_rewards(pre_states, post_states, actions, died_agents)
                
                # Train the DQN (skip during first few years for speed testing)
                if year > 2:  # Only start training after year 2
                    self.env.train_step(pre_states, actions, rewards, post_states)
                
                # Collect metrics
                try:
                    gini = self.netlogo.report('calculate-gini [sugar] of turtles')
                    if gini is None:
                        gini = 0
                except:
                    gini = 0
                
                try:
                    compliance_count = self.netlogo.report(
                        'count turtles with [length compliance-history > 0 and last compliance-history = "full"]'
                    )
                    compliance = compliance_count / max(1, post_population)
                except:
                    compliance = 0
                
                try:
                    evasion_count = self.netlogo.report('count turtles with [strategy = "evade"]')
                    evasion = evasion_count / max(1, post_population)
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
                    'population': post_population,
                    'epsilon': self.env.agent.epsilon,
                    'deaths': len(died_agents)
                })
                
                # Progress update - more frequent for debugging
                if year % 10 == 0 or year < 5:
                    print(f"Year {year}: Pop={post_population}, Deaths={len(died_agents)}, "
                          f"Gini={gini:.2f}, Comply={compliance:.2f}, "
                          f"Epsilon={self.env.agent.epsilon:.4f}")
            
            return episode_results
            
        except Exception as e:
            print(f"Error during episode: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_experiment(self, experiment_name, years=500, audit_rate=0.3, 
                      mode="strict", duration=5, episodes=5):
        """Run complete experiment with DQN agents"""
        experiment_results = []
        os.makedirs('results', exist_ok=True)
        
        print(f"\n🧠 Starting DQN experiment: {experiment_name}")
        print(f"   Parameters: audit={audit_rate}, mode={mode}, duration={duration}")
        print(f"   Neural Network: {self.env.state_size} inputs -> {self.env.total_action_size} outputs")
        
        for episode in range(episodes):
            print(f"\n📊 Episode {episode+1}/{episodes}")
            
            try:
                results = self.run_episode(years, audit_rate, mode, duration)
                
                if len(results) > 0:
                    for result in results:
                        result['episode'] = episode
                        result['experiment'] = experiment_name
                        result['audit_rate'] = audit_rate
                        result['mode'] = mode
                        result['duration'] = duration
                    
                    experiment_results.extend(results)
                    
                    # Save model periodically
                    if (episode + 1) % 10 == 0:
                        self.save_models(experiment_name, episode)
                    
                    print(f"   Episode {episode+1} complete:")
                    print(f"      Final epsilon: {self.env.agent.epsilon:.6f}")
                    print(f"      Memory size: {len(self.env.agent.memory)}")
                    
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        # Save final models
        self.save_models(experiment_name, 'final')
        
        return experiment_results
    
    def save_models(self, experiment_name, suffix):
        """Save DQN model weights"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"results/{experiment_name}_{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            
            self.env.agent.save(f"{dir_name}/unified_dqn_{suffix}.weights.h5")
            
            # Save training parameters
            params = {
                'state_size': self.env.state_size,
                'total_action_size': self.env.total_action_size,
                'movement_action_size': self.env.movement_actions,
                'tax_action_size': self.env.tax_actions,
                'epsilon': float(self.env.agent.epsilon),
                'learning_rate': float(self.env.agent.learning_rate),
                'batch_size': self.env.agent.batch_size,
                'memory_size': len(self.env.agent.memory),
                'suffix': str(suffix)
            }
            
            with open(f"{dir_name}/params_{suffix}.json", 'w') as f:
                json.dump(params, f, indent=2)
                
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def test_connection(self):
        """Test the connection to NetLogo"""
        print("🔄 Testing DQN-NetLogo connection...")
        
        try:
            print("🎯 Running setup...")
            self.netlogo.command('setup')
            print("✅ Setup complete!")
            
            population = int(float(self.netlogo.report('get-population')))
            print(f"👥 Population: {population} turtles")
            
            print("📊 Getting turtle states...")
            states = self.netlogo.report('report-states')
            
            print(f"📝 Got {len(states)} turtle states")
            
            if len(states) > 0:
                processed = self.env.process_netlogo_states(states)
                print(f"🧠 Processed {len(processed)} states for neural network")
                
                # Test action selection
                actions = self.env.choose_actions(processed, is_audit_period=True)
                print(f"[ACTION] Generated {len(actions)} actions")
            
            print("\n🎉 DQN CONNECTION TEST SUCCESSFUL! 🎉")
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """Clean up NetLogo connection"""
        try:
            self.netlogo.kill_workspace()
        except:
            pass


def analyze_dqn_results(results):
    """Analyze and visualize DQN experiment results"""
    if len(results) == 0:
        print("No results generated")
        return None
        
    df = pd.DataFrame(results)
    
    # Group by experiment and year
    summary = df.groupby(['experiment', 'year']).agg({
        'gini': 'mean',
        'compliance_rate': 'mean',
        'evasion_rate': 'mean',
        'total_sugar': 'mean',
        'population': 'mean',
        'epsilon': 'mean',
        'deaths': 'mean'
    }).reset_index()
    
    # Create visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot metrics over time
    for exp in summary.experiment.unique():
        exp_data = summary[summary.experiment == exp]
        
        # Gini coefficient
        axes[0, 0].plot(exp_data.year, exp_data.gini, label=exp)
        axes[0, 0].set_title('Wealth Inequality (Gini Coefficient)')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Gini Coefficient')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Compliance rate
        axes[0, 1].plot(exp_data.year, exp_data.compliance_rate, label=exp)
        axes[0, 1].set_title('Tax Compliance Rate')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Compliance Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Population
        axes[1, 0].plot(exp_data.year, exp_data.population, label=exp)
        axes[1, 0].set_title('Population Over Time')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Deaths per year
        axes[1, 1].plot(exp_data.year, exp_data.deaths, label=exp, alpha=0.7)
        axes[1, 1].set_title('Agent Deaths per Year')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Deaths')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Exploration rate (epsilon)
        axes[2, 0].plot(exp_data.year, exp_data.epsilon, label=exp)
        axes[2, 0].set_title('DQN Exploration Rate (Epsilon)')
        axes[2, 0].set_xlabel('Year')
        axes[2, 0].set_ylabel('Epsilon')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Evasion rate
        axes[2, 1].plot(exp_data.year, exp_data.evasion_rate, label=exp)
        axes[2, 1].set_title('Tax Evasion Rate')
        axes[2, 1].set_xlabel('Year')
        axes[2, 1].set_ylabel('Evasion Rate')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dqn_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary


def main():
    """Main execution function for DQN simulation"""
    netlogo_path = None  # Set this if NetLogo is not in default location
    
    TEST_MODE = True  # Set to False for full experiments
    USE_GUI = False   # Set to True to see NetLogo visualization
    QUICK_TEST = True  # Ultra-fast test mode
    
    try:
        print("="*60)
        print("DEEP Q-NETWORK TAX COMPLIANCE SIMULATION")
        print("="*60)
        print("\n[AI] Using TensorFlow version:", tf.__version__)
        print("[ACTION] Action spaces:")
        print("   - Movement: UP, DOWN, LEFT, RIGHT")
        print("   - Tax compliance: Pay full, Pay partial, Evade")
        print("[PENALTY] Death penalty: Large negative reward for learning")
        
        print("\nInitializing DQN simulation...")
        sim = DQNTaxSimulation(netlogo_path, gui=USE_GUI)
        
        print("\nTesting Python-NetLogo connection...")
        if not sim.test_connection():
            print("Connection test failed. Exiting...")
            sim.close()
            return
        
        print("\n" + "="*60)
        
        if TEST_MODE:
            print("\nRUNNING TEST EXPERIMENT WITH DQN")
            print("-"*40)
            
            if QUICK_TEST:
                experiments = [
                    {
                        'name': 'dqn_quick_test',
                        'audit_rate': 0.5,
                        'mode': 'strict',
                        'duration': 10,
                        'years': 10,  # Very short test - just 10 years
                        'episodes': 1  # Single episode
                    }
                ]
            else:
                experiments = [
                    {
                        'name': 'dqn_test_strict',
                        'audit_rate': 0.5,
                        'mode': 'strict',
                        'duration': 10,
                        'years': 50,  # Reduced from 200 to 50 for faster testing
                        'episodes': 1  # Reduced from 3 to 1 for faster testing
                    }
                ]
        else:
            print("\nRUNNING FULL DQN EXPERIMENTS")
            print("-"*40)
            
            experiments = [
                {
                    'name': 'dqn_strict_low_audit',
                    'audit_rate': 0.3,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 1000,
                    'episodes': 10
                },
                {
                    'name': 'dqn_strict_high_audit',
                    'audit_rate': 0.8,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 1000,
                    'episodes': 10
                },
                {
                    'name': 'dqn_lenient_medium_audit',
                    'audit_rate': 0.5,
                    'mode': 'lenient',
                    'duration': 30,
                    'years': 1000,
                    'episodes': 10
                }
            ]
        
        all_results = []
        
        for i, exp in enumerate(experiments):
            print(f"\n🔬 Experiment {i+1}/{len(experiments)}: {exp['name']}")
            
            results = sim.run_experiment(
                experiment_name=exp['name'],
                years=exp['years'],
                audit_rate=exp['audit_rate'],
                mode=exp['mode'],
                duration=exp['duration'],
                episodes=exp['episodes']
            )
            
            all_results.extend(results)
            print(f"   ✅ Experiment '{exp['name']}' complete!")
        
        print("\n" + "="*60)
        print("ANALYZING DQN RESULTS")
        print("-"*40)
        
        if len(all_results) > 0:
            summary = analyze_dqn_results(all_results)
            
            if summary is not None:
                summary.to_csv('results/dqn_experiment_summary.csv', index=False)
                print("✅ Results saved to results/dqn_experiment_summary.csv")
                print("📊 Plots saved to results/dqn_summary_plot.png")
                
                # Print final statistics
                print("\n📈 Final Statistics:")
                print("-"*40)
                final_year = summary.groupby('experiment').tail(1)
                for _, row in final_year.iterrows():
                    print(f"\n{row['experiment']}:")
                    print(f"  Final Gini: {row['gini']:.3f}")
                    print(f"  Final Compliance: {row['compliance_rate']:.3f}")
                    print(f"  Final Population: {row['population']:.0f}")
                    print(f"  Final Evasion Rate: {row['evasion_rate']:.3f}")
                    print(f"  Avg Deaths/Year: {row['deaths']:.1f}")
        else:
            print("No results generated")
        
        print("\n" + "="*60)
        print("DQN SIMULATION COMPLETE!")
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
        print("2. Install required packages: pip install pynetlogo numpy pandas matplotlib tensorflow")
        print("3. Close any open NetLogo instances")


if __name__ == "__main__":
    main()
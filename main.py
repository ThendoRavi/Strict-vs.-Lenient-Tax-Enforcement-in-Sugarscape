"""
Deep Q-Network (DQN) Multi-Agent Tax Compliance Simulation

This module implements a reinforcement learning framework for simulating tax compliance
behavior in a multi-agent environment. The system uses Deep Q-Networks (DQN) to learn
optimal decision-making strategies for agents navigating a Sugarscape environment while
making tax compliance decisions.

Conceptual Overview:
-------------------
The simulation models a population of agents that must balance resource gathering,
movement, consumption, and tax compliance. Each agent learns through trial and error,
receiving rewards for survival and penalties for death or non-compliance. The DQN
architecture allows agents to develop sophisticated strategies by approximating the
value of different actions in various states.

State Space (9 features):
1. Sugar storage (normalized)
2. Health status (survival viability)
3-4. X and Y position on grid
5. Current punishment status
6-8. Last 3 tax decisions (historical context)
9. Audit cycle phase (temporal awareness)

Temporal Cycle Awareness:
------------------------
The audit cycle phase (feature 9) is a critical component that allows agents to develop
temporally-aware strategies. In the NetLogo simulation, audits occur every 50 ticks
(see 'Sugarscape 2 Constant Growback.nlogo' line: "if ticks mod 50 = 0").

IMPORTANT: Tax actions (indices 6-8) are ONLY valid when is_audit_period=True
(i.e., when phase ≈ 0.0). During other phases, agents can only move and consume.

This allows agents to:
1. Learn temporal patterns: "When I evaded at phase 0.0, I was punished by phase 0.1"
2. Observe consequences: Track how evasion decisions affect survival over full cycle
3. Build decision context: Use phase to remember "I evaded last audit, should I evade again?"
4. Prepare strategically: Accumulate sugar during non-audit periods for taxes at next audit

The correlation with NetLogo:
- Python updates self.env.current_tick = year in run_episode()
- NetLogo tracks ticks and triggers audits via "if ticks mod 50 = 0"
- Both systems stay synchronized through the go() command execution

Key Components:
- DQNAgent: Neural network that learns action values (Q-values)
- MultiAgentDQNEnvironment: Manages state representation and action selection
- DQNTaxSimulation: Coordinates NetLogo simulation with DQN learning
- Analysis functions: Evaluate learning outcomes and compliance patterns

The learning process uses experience replay and target networks to stabilize training,
while epsilon-greedy exploration balances learning new strategies with exploiting
known good actions.
"""

import os
import sys


os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.awt.headless=true -Xmx4g'
# Disable display to force command-line only operation
os.environ['DISPLAY'] = ''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import NetLogo bridge library for agent-based model integration
import pynetlogo

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
from datetime import datetime
import json
import time
import argparse
import logging


class DQNAgent:
    """
    Deep Q-Network Agent for Reinforcement Learning
    
    This class implements a DQN agent that learns to map states to action values.
    The agent uses a neural network to approximate Q-values, which represent the
    expected future reward for taking a specific action in a given state.
    
    Conceptual Framework:
    - Q-learning: Learn optimal action-value function Q(s,a)
    - Deep networks: Approximate Q-function with neural networks
    - Experience replay: Store and reuse past experiences for stable learning
    - Target networks: Use separate network for stable Q-value targets
    - Epsilon-greedy: Balance exploration of new actions vs exploitation of known good actions
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize the DQN agent with network architecture and hyperparameters.
        
        Args:
            state_size: Dimensionality of the state space (number of input features)
            action_size: Number of possible actions the agent can take
            learning_rate: Step size for gradient descent optimization
        """
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize replay memory buffer with maximum capacity
        # Replay memory stores past experiences for training stability
        self.memory = deque(maxlen=10000)
        
        # Set initial exploration rate 
        self.epsilon = 1.0
        # Set minimum exploration rate 
        self.epsilon_min = 0.01
        # Set decay rate for reducing exploration over time
        self.epsilon_decay = 0.999
        
      
        self.learning_rate = learning_rate

        # Set discount factor for future rewards
        self.gamma = 0.95
        # Set mini-batch size for training updates
        self.batch_size = 32
        # Set frequency for updating target network
        self.update_target_every = 100
        # Initialize step counter for tracking training progress
        self.step_count = 0
        
        # Build main Q-network for action-value prediction
        self.q_network = self._build_model()
        # Build target Q-network for stable Q-value targets
        self.target_network = self._build_model()
        # Synchronize target network weights with main network
        self.update_target_network()
        
    def _build_model(self):
        """
        Construct neural network architecture for Q-value approximation.
        
        The network uses multiple fully-connected layers with batch normalization
        and dropout for regularization. The architecture progressively reduces
        dimensionality from state space to action space.
        
        Returns:
            Compiled Keras model ready for training
        """
        model = keras.Sequential([

            layers.Input(shape=(self.state_size,)),
            layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            # Randomly drop 20% of neurons to prevent overfitting
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        # Configure model training parameters
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def update_target_network(self):
        """
        Synchronize target network weights with main Q-network.
        
        The target network provides stable Q-value targets during training.
        Periodic updates prevent the target from changing too rapidly, which
        would destabilize learning.
        """
        # Copy all layer weights from main network to target network
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience tuple in replay memory buffer.
        
        Experience replay breaks temporal correlations in training data by
        randomly sampling past experiences. This improves learning stability.
        
        Args:
            state: Current state observation
            action: Action taken in current state
            reward: Reward received after taking action
            next_state: Resulting state after action
            done: Whether episode terminated after this transition
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon, choose random action (exploration).
        Otherwise, choose action with highest Q-value (exploitation).
        
        Args:
            state: Current state observation (numpy array)
            
        Returns:
            Integer action index
        """

        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: use Q-network to select best action
        state_tensor = tf.expand_dims(state, 0)
        # Compute Q-values for all actions
        q_values = self.q_network(state_tensor, training=False)
        # Return action with maximum Q-value
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Train Q-network on batch of experiences from replay memory.
        
        This implements the core DQN learning algorithm using experience replay
        and the Bellman equation. The network learns to predict Q-values that
        satisfy: Q(s,a) = r + gamma * max_a' Q(s',a')
        
        Training occurs only when sufficient experiences are available.
        """
        # Check if enough experiences collected for batch training
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random mini-batch from replay memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract components from experience tuples into arrays
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Predict current Q-values for actions taken
        current_q_values = self.q_network(states, training=False).numpy()
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_network(next_states, training=False).numpy()
        
        # Update Q-values using Bellman equation for each experience
        for i in range(self.batch_size):
            if dones[i]:
                # Terminal state: Q-value is just the immediate reward
                current_q_values[i][actions[i]] = rewards[i]
            else:
                # Non-terminal: Q-value is reward plus discounted future value
                # Bellman equation: Q(s,a) = r + gamma * max Q(s',a')
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train network to predict updated Q-values
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Decay exploration rate to shift from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Increment training step counter
        self.step_count += 1
        # Periodically update target network for stable learning
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()
    
    def save(self, filepath):
        """
        Persist model weights to disk for later loading.
        
        Args:
            filepath: Path where weights should be saved
        """
        # Save trained Q-network weights to file
        self.q_network.save_weights(filepath)
    
    def load(self, filepath):
        """
        Load previously saved model weights from disk.
        
        Args:
            filepath: Path to saved weight file
        """
        # Load weights into Q-network
        self.q_network.load_weights(filepath)
        # Synchronize target network with loaded weights
        self.update_target_network()


class MultiAgentDQNEnvironment:
    """
    Environment Manager for Multi-Agent DQN Learning
    
    This class bridges the NetLogo simulation environment with the DQN learning
    framework. It handles state normalization, action translation, reward
    calculation, and coordination of multiple learning agents.
    
    Conceptual Framework:
    - State representation: Convert raw simulation data to normalized features
    - Action space: Define legal actions based on simulation constraints
    - Reward shaping: Design reward signals to encourage desired behaviors
    - Multi-agent coordination: Manage learning for population of agents
    """
    
    def __init__(self, n_agents=200):
        """
        Initialize environment with action spaces and learning parameters.
        
        Args:
            n_agents: Number of agents in simulation (default 200)
        """
        # Store population size
        self.n_agents = n_agents

        self.state_size = 9
        
        # Define action space components
        # UP, DOWN, LEFT, RIGHT
        self.movement_actions = 4  
        
        # Consumption actions: resource gathering decisions
        # CONSUME, HARVEST
        self.consumption_actions = 2  
        
        # Tax compliance actions: payment decisions during audits
        # PAY_FULL, PAY_PARTIAL, EVADE  
        self.tax_actions = 3  
        
        # Calculate total action space size
        self.total_action_size = self.movement_actions + self.consumption_actions + self.tax_actions
        
        # Create unified DQN agent for all actions - Parameter Sharing across agents
        self.agent = DQNAgent(self.state_size, self.total_action_size)
        
        # Initialize tracking structures for agent histories
        self.agent_states = {}
        self.agent_histories = {}
        
        # Define large negative reward for agent death
        self.death_penalties = -100
        
        # Initialize simulation tick counter
        self.current_tick = 0
        
        # Set audit frequency (tax decisions every N ticks)
        # 50 ticks is equivalent to one year
        self.audit_frequency = 50
        
        # Create mapping from action indices to action names
        self.action_mapping = {

            0: 'UP',
            1: 'DOWN', 
            2: 'LEFT',
            3: 'RIGHT',

            4: 'CONSUME',
            5: 'HARVEST',

            6: 'PAY_FULL',
            7: 'PAY_PARTIAL',
            8: 'EVADE'
        }
        
    def normalize_state(self, sugar_level, x_pos, y_pos, punished, punishment_history, metabolism=2):
        """
        Transform raw simulation state into normalized neural network input.
        
        State Components:
        1. Sugar storage: Total accumulated resources
        2. Health status: Resources relative to survival needs
        3. Spatial position: X and Y coordinates on grid
        4. Punishment status: Currently under punishment or not
        5. Historical decisions: Past 3 tax choices
        6. Audit cycle phase: Temporal awareness for strategic planning
        
        
        The temporal awareness helps agents learn:
        1. Whether recent evasion/compliance led to good/bad outcomes
        2. If they have enough sugar reserves to risk evasion at next audit
        3. Whether punishment from last audit is still active
        4. Patterns: "I evaded at phase 0.0, got punished, now at phase 0.5 suffering"
        
        Args:
            sugar_level: Current sugar reserves of agent
            x_pos: X coordinate on simulation grid
            y_pos: Y coordinate on simulation grid
            punished: Boolean punishment status
            punishment_history: List of past tax decisions
            metabolism: Agent's resource consumption rate
            
        Returns:
            Normalized state vector (numpy array) with 9 features
        """
        # Normalize sugar storage to [0,1] range
        max_expected_sugar = 100.0
        sugar_storage = min(sugar_level / max_expected_sugar, 1.0)
        
        # Calculate health status relative to survival threshold
        # Health critical when reserves below 5 turns of metabolism
        survival_threshold = metabolism * 5
        health_status = min(sugar_level / survival_threshold, 1.0) if survival_threshold > 0 else 1.0
        
        # Normalize grid positions to [0,1] range
        # Simulation uses 50x50 grid (indices 0-49)
        norm_x = x_pos / 49.0
        norm_y = y_pos / 49.0
        
        # Convert boolean punishment status to float
        punishment_flag = float(punished)
        
        # Encode last 3 tax decisions as normalized values
        history_encoding = np.zeros(3)
        for i, hist_idx in enumerate([-1, -2, -3]):
            # Check if history contains entry at this index
            if len(punishment_history) >= abs(hist_idx):
                action = punishment_history[hist_idx]
                # Normalize action values (0,1,2) to [0,0.5,1] range
                history_encoding[i] = action / 2.0 if action >= 0 else 0.0
        
        # Calculate audit cycle phase for temporal awareness
        # Returns value in [0, 1) indicating position in 50-tick audit cycle
        audit_phase = (self.current_tick % self.audit_frequency) / self.audit_frequency
        
        # Construct complete state vector with 9 features
        state = np.array([
            sugar_storage,          # Feature 0: Raw resource amount (0-1 normalized)
            health_status,          # Feature 1: Survival viability (0-1 normalized)
            norm_x,                 # Feature 2: X position (0-1 normalized)
            norm_y,                 # Feature 3: Y position (0-1 normalized)
            punishment_flag,        # Feature 4: Current punishment (0 or 1)
            history_encoding[0],    # Feature 5: Most recent tax decision (0-1)
            history_encoding[1],    # Feature 6: Second most recent decision (0-1)
            history_encoding[2],    # Feature 7: Third most recent decision (0-1)
            audit_phase             # Feature 8: Position in audit cycle (0-1)
        ])
        
        return state
    
    def process_netlogo_states(self, raw_states):
        """
        Convert raw NetLogo simulation states to normalized network inputs.
        
        This method processes the state information received from the NetLogo
        simulation and transforms it into the format required by the neural
        network. It also maintains agent history for temporal decision context.
        
        Args:
            raw_states: List of state arrays from NetLogo simulation
            
        Returns:
            Dictionary mapping agent IDs to processed state information
        """
        # Initialize dictionary to store processed states
        processed_states = {}
        
        # Process each agent's state information
        for state_data in raw_states:
            # Verify state data contains minimum required elements
            if len(state_data) >= 8:
                # Agent unique identifier
                turtle_id = int(state_data[0])
                # Current resource level      
                sugar_level = state_data[1]
                # Punishment flag         
                punished = state_data[2]
                # Length of decision history            
                history_len = state_data[3]
                # Most recent action taken       
                last_action = state_data[4]
                # # Total punishments received         
                punishment_count = state_data[5]    
                
                # Extract spatial coordinates if available (extended state format)
                if len(state_data) >= 10:
                    # Historical evasion success
                    evasion_success_rate = state_data[6]
                    # Compliance behavior pattern    
                    compliance_pattern = state_data[7]
                    # X coordinate      
                    x_pos = state_data[8]
                    # Y coordinate                   
                    y_pos = state_data[9]                   
                else:
                    # Fallback to random positions for older state format
                    x_pos = np.random.randint(0, 50)
                    y_pos = np.random.randint(0, 50)
                
                # Initialize agent history tracking if first observation
                if turtle_id not in self.agent_histories:
                    self.agent_histories[turtle_id] = []
                
                # Update agent's decision history
                if last_action >= 0:
                    # Append new action to history
                    self.agent_histories[turtle_id].append(last_action)
                    # Maintain only last 3 actions for memory efficiency
                    if len(self.agent_histories[turtle_id]) > 3:
                        self.agent_histories[turtle_id] = self.agent_histories[turtle_id][-3:]
                
                # Normalize state for neural network input
                normalized_state = self.normalize_state(
                    sugar_level, x_pos, y_pos, punished, 
                    self.agent_histories.get(turtle_id, [])
                )
                
                # Store processed state with metadata
                processed_states[turtle_id] = {
                    # Normalized state vector
                    'state': normalized_state,      
                    # Raw sugar level
                    'sugar_level': sugar_level,     
                    # Punishment status
                    'punished': punished,           
                    # Original state data
                    'raw_data': state_data          
                }
                
        return processed_states
    
    def get_legal_actions(self, is_audit_period=False, is_punished=False):
        """
        Determine which actions are legal in current simulation state.
        
        Not all actions are valid at all times. Movement may be restricted
        during punishment, and tax decisions are only relevant during audits.
        This method creates a boolean mask indicating legal actions.
        
        Args:
            is_audit_period: Whether current tick is an audit period
            is_punished: Whether agent is currently under punishment
            
        Returns:
            Boolean array mask (True = legal action)
        """
        # Initialize mask with all actions illegal
        legal_mask = np.zeros(self.total_action_size, dtype=bool)
        
        # Movement actions legal unless agent is punished
        if not is_punished:
            # Enable UP, DOWN, LEFT, RIGHT
            legal_mask[0:4] = True  
        
        # Consumption/harvest actions always legal
        legal_mask[4:6] = True  
        
        # Tax compliance actions only legal during audit periods
        if is_audit_period:
            # Enable PAY_FULL, PAY_PARTIAL, EVADE
            legal_mask[6:9] = True  
        
        return legal_mask

    def choose_actions(self, processed_states, is_audit_period=False):
        """
        Select actions for all agents using DQN with legal action masking.
        
        This method applies the learned policy to choose actions for each agent
        while respecting legal action constraints. It uses batch processing for
        computational efficiency when handling many agents.
        
        Args:
            processed_states: Dictionary of agent IDs to state information
            is_audit_period: Whether current tick allows tax decisions
            
        Returns:
            Dictionary mapping agent IDs to selected action indices
        """
        # Initialize action dictionary
        actions = {}
        
        # Return empty if no states provided
        if not processed_states:
            return actions
        
        # Extract agent IDs and prepare for batch processing
        turtle_ids = list(processed_states.keys())
        # Stack all states into single numpy array for efficient batch prediction
        states_batch = np.array([processed_states[tid]['state'] for tid in turtle_ids])
        # Extract punishment status for each agent
        punished_batch = [bool(processed_states[tid]['punished']) for tid in turtle_ids]
        
        # Get Q-values for all states in single forward pass (efficient)
        q_values_batch = self.agent.q_network.predict(states_batch, verbose=0)
        
        # Process each agent's Q-values to select legal action
        for i, turtle_id in enumerate(turtle_ids):
            # Get agent's punishment status
            is_punished = punished_batch[i]
            # Get agent's Q-values for all actions
            q_values = q_values_batch[i]
            
            # Determine which actions are legal for this agent
            legal_mask = self.get_legal_actions(is_audit_period, is_punished)
            
            # Mask illegal actions by setting their Q-values to negative infinity
            masked_q_values = np.where(legal_mask, q_values, -np.inf)
            
            # Apply epsilon-greedy policy with legal action constraints
            if np.random.random() <= self.agent.epsilon:
                # Exploration: randomly select from legal actions only
                legal_actions = np.where(legal_mask)[0]
                action = np.random.choice(legal_actions)
            else:
                # Exploitation: select legal action with highest Q-value
                action = np.argmax(masked_q_values)
            
            # Store selected action for this agent
            actions[turtle_id] = action
            
        return actions
    
    def translate_actions_to_netlogo(self, actions, is_audit_period=False):
        """
        Convert DQN action indices to NetLogo-compatible command format.
        
        The DQN outputs action indices, but NetLogo requires specific command
        formats. This method translates between the two representations,
        separating movement and tax decisions for different NetLogo handlers.
        
        Args:
            actions: Dictionary mapping agent IDs to action indices
            is_audit_period: Whether tax decisions should be processed
            
        Returns:
            Tuple of (movement_commands, tax_decisions) lists
        """
        # Initialize command lists
        movement_commands = []
        tax_decisions = []
        
        # Process each agent's action
        for turtle_id, action in actions.items():
            # Handle movement actions (indices 0-3)
            if action in [0, 1, 2, 3]:
                # Convert action index to direction string
                direction = self.action_mapping[action]
                # Store as (agent_id, direction) tuple
                movement_commands.append((turtle_id, direction))
            
            # Handle tax compliance actions (indices 6-8), only during audits
            elif action in [6, 7, 8] and is_audit_period:
                # PAY_FULL
                if action == 6:  
                    # NetLogo uses 0 for full payment
                    tax_decisions.append((turtle_id, 0))
                    # PAY_PARTIAL
                elif action == 7:  
                    # NetLogo uses 1 for partial payment
                    tax_decisions.append((turtle_id, 1))
                    # EVADE
                elif action == 8:  
                    # NetLogo uses 2 for evasion
                    tax_decisions.append((turtle_id, 2))
            
            # Consumption/harvest actions (4-5) handled automatically by NetLogo
        
        return movement_commands, tax_decisions
    
    def calculate_rewards(self, pre_states, post_states, actions, died_agents):
        """
        Compute reward signals for reinforcement learning.
        
        Reward shaping is critical for learning. This implementation uses
        sparse rewards focused on survival, with large negative penalties
        for death to encourage risk-averse behavior.
        
        Reward Structure:
        - Most actions: 0 reward (neutral)
        - Agent death: -100 reward (strong discouragement)
        
        Args:
            pre_states: Agent states before action execution
            post_states: Agent states after action execution
            actions: Actions taken by each agent
            died_agents: Set of agent IDs that died this step
            
        Returns:
            Dictionary mapping agent IDs to reward values
        """
        # Initialize rewards dictionary
        rewards = {}
        
        # Calculate reward for each agent
        for turtle_id in pre_states:
            if turtle_id in died_agents:
                # Agent died: apply large negative penalty
                rewards[turtle_id] = self.death_penalties
            elif turtle_id in post_states:
                # Agent survived: calculate standard reward
                pre_punished = pre_states[turtle_id]['punished']
                post_punished = post_states[turtle_id]['punished']
                # Get action taken by agent
                action = actions.get(turtle_id, 0)
                
                reward = 0.0
                rewards[turtle_id] = reward
            else:
                # Agent disappeared without being in death list
                rewards[turtle_id] = self.death_penalties
                
        return rewards
    
    def train_step(self, pre_states, actions, rewards, post_states):
        """
        Execute single training step for DQN agent.
        
        This method stores experiences in replay memory and triggers network
        training. Each experience consists of (state, action, reward, next_state, done)
        which allows the agent to learn from consequences of its actions.
        
        Args:
            pre_states: States before actions
            actions: Actions taken
            rewards: Rewards received
            post_states: States after actions
        """
        # Process each agent's experience
        for turtle_id in pre_states:
            # Only process if agent took an action
            if turtle_id in actions:
                # Extract state before action
                pre_state = pre_states[turtle_id]['state']
                # Extract action taken
                action = actions[turtle_id]
                # Extract reward received
                reward = rewards.get(turtle_id, 0)
                # Determine if episode ended (death or disappeared)
                done = turtle_id not in post_states or reward == self.death_penalties
                
                if done:
                    # Episode ended: use zero vector as next state
                    next_state = np.zeros(self.state_size)
                else:
                    # Episode continues: use actual next state
                    next_state = post_states[turtle_id]['state']
                
                # Store experience tuple in replay memory
                self.agent.remember(pre_state, action, reward, next_state, done)
        
        # Trigger neural network training on batch of experiences
        # This updates Q-value predictions based on Bellman equation
        self.agent.replay()


class DQNTaxSimulation:
    """
    Main Simulation Coordinator
    
    This class manages the overall simulation workflow, coordinating between
    the NetLogo agent-based model and the DQN learning system. It handles
    initialization, episode execution, result collection, and model persistence.
    
    Conceptual Flow:
    1. Initialize NetLogo simulation and DQN environment
    2. For each episode:
       a. Run simulation ticks
       b. Collect agent states
       c. Select actions using DQN
       d. Execute actions in NetLogo
       e. Calculate rewards
       f. Train DQN on experiences
    3. Analyze results and save models
    """
    
    def __init__(self, netlogo_path=None, gui=False):
        """
        Initialize simulation with NetLogo connection.
        
        Args:
            netlogo_path: Path to NetLogo installation directory
            gui: Whether to display NetLogo GUI (forced False on clusters)
        """
        # Force headless mode for cluster compatibility
        self.gui = False
        
        # Ensure Java runs in headless mode without GUI
        os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.awt.headless=true -Xmx4g'
        os.environ['DISPLAY'] = ''
        
        try:
            # Log connection initialization
            print("Initializing NetLogo connection...")
            print(f"   Path: {netlogo_path}")
            print(f"   GUI mode: FORCED FALSE (headless)")
            print(f"   Java options: {os.environ.get('JAVA_TOOL_OPTIONS', 'not set')}")
            
            if netlogo_path:
                # Attempt connection with explicit version parameter
                try:
                    # Method 1: Specify NetLogo version explicitly
                    self.netlogo = pynetlogo.NetLogoLink(
                        gui=False,                      
                        netlogo_home=netlogo_path,      
                        netlogo_version='6.4',          
                        jvm_home=None                   
                    )
                    print("Connected using netlogo_version parameter")
                except TypeError:
                    # Method 2: Fallback without version parameter for older PyNetLogo
                    print("Trying alternative connection method...")
                    self.netlogo = pynetlogo.NetLogoLink(
                        gui=False,
                        netlogo_home=netlogo_path
                    )
                    print("Connected using basic parameters")
            else:
                # Use default NetLogo location if no path specified
                self.netlogo = pynetlogo.NetLogoLink(gui=False)
            
            # Specify NetLogo model file name
            model_file = 'Sugarscape 2 Constant Growback.nlogo'
            # Verify model file exists in current directory
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"NetLogo model file '{model_file}' not found in {os.getcwd()}")
            
            # Load the NetLogo model into the workspace
            print(f"Loading NetLogo model: {model_file}")
            self.netlogo.load_model(model_file)
            print("NetLogo model loaded successfully")
            
            # Initialize DQN environment manager
            self.env = MultiAgentDQNEnvironment()
            # Initialize results storage
            self.results = []
            
        except Exception as e:
            # Log detailed error information for debugging
            print(f"Error initializing NetLogo: {e}")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   NetLogo path: {netlogo_path}")
            print(f"   Files in directory: {os.listdir('.')[:10]}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_episode(self, years, audit_rate, mode, duration):
        """
        Execute single simulation episode with DQN learning.
        
        An episode consists of running the simulation for a specified number
        of years while agents learn optimal policies through trial and error.
        Each year involves state observation, action selection, simulation
        execution, reward calculation, and network training.
        
        Args:
            years: Number of simulation years to run
            audit_rate: Probability of audit enforcement
            mode: Enforcement mode (strict/lenient)
            duration: Punishment duration in ticks
            
        Returns:
            List of results dictionaries (one per year)
        """
        try:
            # Log episode start with parameters
            print(f"\nStarting episode: {years} years, audit_rate={audit_rate}, mode={mode}, duration={duration}")
            
            # Configure NetLogo simulation parameters
            self.netlogo.command(f'set-params {audit_rate} "{mode}" {duration}')
            # Initialize NetLogo simulation (creates agents, sets up environment)
            self.netlogo.command('setup')
            
            # Initialize episode tracking variables
            episode_results = []
            consecutive_empty_years = 0
            
            # Track episode statistics for analysis
            total_deaths = 0
            total_rewards = 0
            action_counts = {'movement': 0, 'consumption': 0, 'tax': 0}
            
            # Main simulation loop (one iteration per year)
            for year in range(years):
                # Update environment's internal tick counter
                self.env.current_tick = year
                
                # Retrieve current state of all agents from NetLogo
                raw_states = self.netlogo.report('report-states')
                # Get current population count
                population = int(float(self.netlogo.report('get-population')))
                
                # Check for population extinction
                if len(raw_states) == 0 or population == 0:
                    print(f"No turtles remaining at year {year}")
                    break
                    
                
                # Reset extinction counter when population exists
                consecutive_empty_years = 0
                
                # Process raw states into normalized neural network inputs
                pre_states = self.env.process_netlogo_states(raw_states)
                
                # Determine if current year is an audit period
                # Audits occur periodically based on audit_frequency
                is_audit_period = (year % self.env.audit_frequency == 0)
                
                # Log audit periods for debugging
                if is_audit_period:
                    print(f"  Year {year}: AUDIT PERIOD - agents making tax decisions")
                elif year % 25 == 0:
                    print(f"  Year {year}: Regular period - movement and consumption only")
                
                # Select actions for all agents using DQN policy
                start_time = time.time()
                
                actions = self.env.choose_actions(pre_states, is_audit_period)
                
                action_time = time.time() - start_time
                
                # Track action type distribution for analysis
                movement_count = sum(1 for a in actions.values() if a in [0, 1, 2, 3])
                consumption_count = sum(1 for a in actions.values() if a in [4, 5])
                tax_count = sum(1 for a in actions.values() if a in [6, 7, 8])
                
                # Accumulate action counts
                action_counts['movement'] += movement_count
                action_counts['consumption'] += consumption_count
                action_counts['tax'] += tax_count
                
                # Log action distribution periodically
                if year % 25 == 0 or is_audit_period:
                    print(f"    Actions: {movement_count} move, {consumption_count} consume, {tax_count} tax (took {action_time:.3f}s)")
                
                # Convert DQN actions to NetLogo command format
                movement_commands, tax_decisions = self.env.translate_actions_to_netlogo(actions, is_audit_period)
                
                # Send movement commands to NetLogo if any agents moved
                if movement_commands:
                    # Format as NetLogo list: [[id1 "DIR1"] [id2 "DIR2"] ...]
                    movement_string = "["
                    for turtle_id, direction in movement_commands:
                        movement_string += f"[{turtle_id} \"{direction}\"] "
                    movement_string += "]"
                    # Execute NetLogo command to process movements
                    self.netlogo.command(f'receive-movement-commands {movement_string}')
                
                # Send tax decisions to NetLogo during audit periods
                if is_audit_period and tax_decisions:
                    # Convert tax decisions to action list indexed by agent
                    tax_action_list = [0] * population  # Default to full payment
                    # Get list of agent IDs from raw states
                    turtle_list = [int(state[0]) for state in raw_states]
                    
                    # Map tax decisions to correct agent positions
                    for turtle_id, tax_action in tax_decisions:
                        if turtle_id in turtle_list:
                            idx = turtle_list.index(turtle_id)
                            if idx < len(tax_action_list):
                                tax_action_list[idx] = tax_action
                    
                    # Format as NetLogo list and send
                    action_string = "[" + " ".join(map(str, tax_action_list)) + "]"
                    self.netlogo.command(f'receive-actions {action_string}')
                
                # Execute one simulation step in NetLogo
                # This applies actions and runs simulation logic
                self.netlogo.command('go')
                
                # Retrieve post-action states from NetLogo
                post_raw_states = self.netlogo.report('report-states')
                rewards_after = self.netlogo.report('report-rewards')
                post_population = int(float(self.netlogo.report('get-population')))
                
                # Process post-action states
                post_states = self.env.process_netlogo_states(post_raw_states)
                
                # Identify agents that died during this step
                # Agents in pre_states but not in post_states have died
                died_agents = set(pre_states.keys()) - set(post_states.keys())
                
                # Calculate reward signals for learning
                rewards = self.env.calculate_rewards(pre_states, post_states, actions, died_agents)
                
                # Track reward statistics
                episode_reward_sum = sum(rewards.values())
                death_count = len(died_agents)
                total_deaths += death_count
                total_rewards += episode_reward_sum
                
                # Separate punishment rewards from death penalties
                punishment_rewards = sum(r for r in rewards.values() if r < 0 and r > -100)
                death_rewards = sum(r for r in rewards.values() if r == -100)
                
                # Log significant reward events
                if year % 25 == 0 or death_count > 0 or abs(episode_reward_sum) > 50:
                    print(f"    Rewards: total={episode_reward_sum:.1f}, deaths={death_count}, punishment={punishment_rewards:.1f}")
                
                # Train DQN on collected experiences
                memory_before = len(self.env.agent.memory)
                epsilon_before = self.env.agent.epsilon
                self.env.train_step(pre_states, actions, rewards, post_states)
                
                # Log training progress periodically
                if year % 50 == 0 and year > 0:
                    print(f"    Training: memory={len(self.env.agent.memory)}, epsilon={epsilon_before:.4f}, "
                          f"added {len(self.env.agent.memory) - memory_before} experiences")
                
                # Collect simulation metrics for analysis
                try:
                    # Calculate Gini coefficient for wealth inequality
                    gini = self.netlogo.report('calculate-gini [sugar] of turtles')
                    if gini is None:
                        gini = 0
                except:
                    gini = 0
                
                # Calculate compliance rate from agent histories
                try:
                    # Count agents with any compliance history
                    agents_with_history = self.netlogo.report('count turtles with [length compliance-history > 0]')
                    if agents_with_history > 0:
                        # Count agents whose last action was full payment
                        compliance_count = self.netlogo.report(
                            'count turtles with [length compliance-history > 0 and last compliance-history = "full"]'
                        )
                        compliance = compliance_count / max(1, post_population)
                    else:
                        compliance = 0
                except:
                    compliance = 0
                
                # Calculate evasion rate from agent histories
                try:
                    # Count agents with any compliance history
                    agents_with_history = self.netlogo.report('count turtles with [length compliance-history > 0]')
                    if agents_with_history > 0:
                        # Count agents whose last action was no payment
                        evasion_count = self.netlogo.report(
                            'count turtles with [length compliance-history > 0 and last compliance-history = "none"]'
                        )
                        evasion = evasion_count / max(1, post_population)
                    else:
                        evasion = 0
                except:
                    evasion = 0
                
                # Calculate partial payment rate from agent histories
                try:
                    # Count agents with any compliance history
                    agents_with_history = self.netlogo.report('count turtles with [length compliance-history > 0]')
                    if agents_with_history > 0:
                        # Count agents whose last action was partial payment
                        partial_count = self.netlogo.report(
                            'count turtles with [length compliance-history > 0 and last compliance-history = "partial"]'
                        )
                        partial_payment = partial_count / max(1, post_population)
                    else:
                        partial_payment = 0
                except:
                    partial_payment = 0

                # Calculate total sugar in economy
                try:
                    total_sugar = self.netlogo.report('sum [sugar] of turtles')
                    if total_sugar is None:
                        total_sugar = 0
                except:
                    total_sugar = 0
                
                # Store year's results
                episode_results.append({
                    'year': year,
                    'gini': gini,
                    'compliance_rate': compliance,
                    'evasion_rate': evasion,
                    'partial_payment_rate': partial_payment,
                    'total_sugar': total_sugar,
                    'population': post_population,
                    'epsilon': self.env.agent.epsilon,
                    'deaths': len(died_agents)
                })
                
                # Debug logging at key intervals to verify continuous tracking
                if year in [0, 1, 2, 10, 20, 30, 49, 50, 51, 100]:
                    print(f"  [Year {year}] Pop={post_population}, Gini={gini:.3f}, Sugar={total_sugar:.0f}, Deaths={len(died_agents)}")
                
                # Print periodic progress summary
                if year % 50 == 0:
                    avg_reward = total_rewards / max(1, year + 1)
                    print(f"\nYear {year} Summary:")
                    print(f"    Population: {post_population} (deaths this year: {len(died_agents)})")
                    print(f"    Economics: Gini={gini:.3f}, Total Sugar={total_sugar:.0f}")
                    print(f"    Tax Behavior: Compliance={compliance:.1%}, Evasion={evasion:.1%}")
                    print(f"    Learning: Epsilon={self.env.agent.epsilon:.4f}, Memory={len(self.env.agent.memory)}")
                    print(f"    Rewards: Current={episode_reward_sum:.1f}, Average={avg_reward:.2f}")
                    print(f"    Actions so far: {action_counts['movement']} move, {action_counts['consumption']} consume, {action_counts['tax']} tax")
            
            # Print episode completion summary
            print(f"\nEpisode Complete")
            print(f"    Total years: {len(episode_results)}")
            print(f"    Data points collected: {len(episode_results)} (tracking every year)")
            print(f"    Total deaths: {total_deaths}")
            print(f"    Average reward per year: {total_rewards / max(1, len(episode_results)):.2f}")
            print(f"    Final epsilon: {self.env.agent.epsilon:.6f}")
            print(f"    Final memory size: {len(self.env.agent.memory)}")
            
            return episode_results
            
        except Exception as e:
            # Log any errors that occur during episode
            print(f"Error during episode: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_experiment(self, experiment_name, years=500, audit_rate=0.3, 
                      mode="strict", duration=5, episodes=5):
        """
        Execute complete multi-episode experiment.
        
        An experiment consists of multiple episodes to gather statistical
        evidence about learned policies. Results are aggregated and models
        are periodically saved.
        
        Args:
            experiment_name: Identifier for this experiment run
            years: Years per episode
            audit_rate: Audit probability parameter
            mode: Enforcement mode
            duration: Punishment duration
            episodes: Number of episodes to run
            
        Returns:
            List of all episode results combined
        """
        # Initialize experiment results storage
        experiment_results = []
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Log experiment configuration
        print(f"\nStarting DQN experiment: {experiment_name}")
        print(f"   Parameters: audit={audit_rate}, mode={mode}, duration={duration}")
        print(f"   Neural Network: {self.env.state_size} inputs (9 features) -> {self.env.total_action_size} outputs")
        print(f"   State features: sugar, health, x, y, punishment, last 3 tax decisions, audit phase")
        print(f"   Action Space: 0-3=Move, 4-5=Consume, 6-8=Tax(Pay/Partial/Evade)")
        print(f"   Death penalty: {self.env.death_penalties}")
        print(f"   Audit frequency: every {self.env.audit_frequency} ticks (synchronized with NetLogo)")
        print(f"   Temporal awareness: Agents know position in 50-tick audit cycle")
        
        # Record experiment start time
        experiment_start_time = time.time()
        
        # Execute each episode
        for episode in range(episodes):
            episode_start_time = time.time()
            # Log episode start
            print(f"\n" + "="*60)
            print(f"Episode {episode+1}/{episodes} - Starting at {datetime.now().strftime('%H:%M:%S')}")
            print(f"="*60)
            
            try:
                # Run single episode
                results = self.run_episode(years, audit_rate, mode, duration)
                episode_time = time.time() - episode_start_time
                
                # Process results if episode completed successfully
                if len(results) > 0:
                    # Add experiment metadata to each result
                    for result in results:
                        result['episode'] = episode
                        result['experiment'] = experiment_name
                        result['audit_rate'] = audit_rate
                        result['mode'] = mode
                        result['duration'] = duration
                    
                    # Append to experiment results
                    experiment_results.extend(results)
                    
                    # Save model checkpoint periodically
                    if (episode + 1) % 10 == 0:
                        print(f"Saving model checkpoint at episode {episode+1}...")
                        self.save_models(experiment_name, episode)
                    
                    # Log episode completion
                    print(f"\nEpisode {episode+1} COMPLETE (took {episode_time/60:.1f} minutes)")
                    print(f"    Final learning state:")
                    print(f"      - Epsilon (exploration): {self.env.agent.epsilon:.6f}")
                    print(f"      - Memory size: {len(self.env.agent.memory):,} experiences")
                    print(f"      - Years completed: {len(results)}")
                    
                    # Log final episode statistics
                    if results:
                        final_pop = results[-1]['population']
                        final_gini = results[-1]['gini']
                        final_compliance = results[-1]['compliance_rate']
                        print(f"      - Final population: {final_pop}")
                        print(f"      - Final Gini coefficient: {final_gini:.3f}")
                        print(f"      - Final compliance rate: {final_compliance:.1%}")
                    
            except Exception as e:
                # Log episode errors but continue experiment
                print(f"Error in episode {episode+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save final trained models
        print(f"\nSaving final models...")
        self.save_models(experiment_name, 'final')
        
        # Calculate total experiment duration
        experiment_time = time.time() - experiment_start_time
        
        # Print comprehensive experiment summary
        print(f"\n" + "="*60)
        print(f"EXPERIMENT '{experiment_name}' COMPLETE")
        print(f"="*60)
        print(f"Total time: {experiment_time/60:.1f} minutes ({experiment_time/3600:.2f} hours)")
        print(f"Episodes completed: {len([r for r in experiment_results if r])}")
        print(f"Final learning parameters:")
        print(f"    - Epsilon: {self.env.agent.epsilon:.6f}")
        print(f"    - Memory: {len(self.env.agent.memory):,} experiences")
        print(f"    - Network updates: {self.env.agent.step_count}")
        
        # Calculate aggregate statistics if results available
        if experiment_results:
            total_years = len(experiment_results)
            avg_pop = sum(r['population'] for r in experiment_results) / total_years
            avg_gini = sum(r['gini'] for r in experiment_results) / total_years
            avg_compliance = sum(r['compliance_rate'] for r in experiment_results) / total_years
            total_deaths = sum(r['deaths'] for r in experiment_results)
            
            print(f"Experiment Statistics:")
            print(f"    - Total simulation years: {total_years:,}")
            print(f"    - Average population: {avg_pop:.1f}")
            print(f"    - Average Gini coefficient: {avg_gini:.3f}")
            print(f"    - Average compliance rate: {avg_compliance:.1%}")
            print(f"    - Total agent deaths: {total_deaths:,}")
        
        return experiment_results
    
    def save_models(self, experiment_name, suffix):
        """
        Persist trained model weights and parameters to disk.
        
        Args:
            experiment_name: Identifier for experiment
            suffix: Suffix to append to filename (episode number or 'final')
        """
        try:
            # Create timestamped directory for this save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"results/{experiment_name}_{timestamp}"
            os.makedirs(dir_name, exist_ok=True)
            
            # Save neural network weights
            self.env.agent.save(f"{dir_name}/unified_dqn_{suffix}.weights.h5")
            
            # Prepare hyperparameter dictionary
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
            
            # Save parameters as JSON file
            with open(f"{dir_name}/params_{suffix}.json", 'w') as f:
                json.dump(params, f, indent=2)
                
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def test_connection(self):
        """
        Verify NetLogo-DQN integration is functioning correctly.
        
        This method runs a basic test to ensure the simulation can initialize,
        retrieve states, select actions, and process results properly.
        
        Returns:
            Boolean indicating test success
        """
        print("Testing DQN-NetLogo connection...")
        
        try:
            # Initialize NetLogo simulation
            print("Running setup...")
            self.netlogo.command('setup')
            print("Setup complete")
            
            # Get initial population count
            population = int(float(self.netlogo.report('get-population')))
            print(f"Population: {population} turtles")
            
            # Retrieve agent states
            print("Getting turtle states...")
            states = self.netlogo.report('report-states')
            
            # Verify states were retrieved
            if states is not None and len(states) > 0:
                print(f"Got states for {len(states)} agents")
                
            # Process states for neural network
            if states is not None and len(states) > 0:
                processed = self.env.process_netlogo_states(states)
                print(f"Processed {len(processed)} states for neural network")
                
                # Examine sample state
                if processed:
                    sample_state_dict = next(iter(processed.values()))
                    sample_state = sample_state_dict['state']
                    print(f"Sample state shape: {sample_state.shape}")
                    print(f"Sample state values: {sample_state[:3]}...")
                
                # Test action selection
                actions = self.env.choose_actions(processed, is_audit_period=True)
                print(f"Generated {len(actions)} actions")
                
                # Analyze action distribution
                if actions:
                    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CONSUME', 'NO_CONSUME', 'PAY_FULL', 'PAY_PARTIAL', 'EVADE']
                    action_counts = {}
                    for action in actions.values():
                        action_name = action_names[action] if action < len(action_names) else f'UNKNOWN_{action}'
                        action_counts[action_name] = action_counts.get(action_name, 0) + 1
                    print(f"Action distribution: {action_counts}")
            
            # Test successful
            print("\nDQN CONNECTION TEST SUCCESSFUL")
            return True
            
        except Exception as e:
            # Test failed
            print(f"Connection test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def close(self):
        """
        Clean up resources and terminate connections.
        
        Proper cleanup is essential for releasing threads, closing JVM,
        and ensuring files are properly saved before exit.
        """
        # Clear TensorFlow computational graph and release GPU memory
        try:
            print("Clearing TensorFlow sessions...")
            import tensorflow as tf
            from tensorflow import keras
            
            # Clear Keras backend session
            keras.backend.clear_session()
            
            # Delete model references to free memory
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'target_model'):
                del self.target_model
            
            print("TensorFlow cleanup complete")
        except Exception as e:
            print(f"Warning during TensorFlow cleanup: {e}")
        
        # Close all matplotlib figure windows
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception as e:
            print(f"Warning during matplotlib cleanup: {e}")
        
        # Close NetLogo connection
        try:
            print("Closing NetLogo connection...")
            self.netlogo.kill_workspace()
            print("NetLogo workspace closed")
        except Exception as e:
            print(f"Warning during workspace cleanup: {e}")
        
        # Skip JVM shutdown as it can hang
        # System exit will handle JVM termination
        print("Skipping JVM shutdown (will force exit)...")
        
        # Brief pause to ensure file operations complete
        import time
        time.sleep(0.2)


def analyze_dqn_results(results):
    """
    Analyze and visualize experimental results.
    
    This function aggregates results across episodes, computes summary
    statistics, and generates visualization plots for key metrics.
    
    Args:
        results: List of result dictionaries from all episodes
        
    Returns:
        DataFrame with aggregated summary statistics
    """
    # Check if results available
    if len(results) == 0:
        print("No results generated")
        return None
    
    print(f"\nAnalyzing {len(results)} total data points...")
    
    # Convert results list to pandas DataFrame
    df = pd.DataFrame(results)
    
    # Log data distribution for verification
    print(f"   Data spans {df['year'].min()} to {df['year'].max()} years")
    print(f"   Unique years tracked: {df['year'].nunique()}")
    print(f"   Experiments: {df['experiment'].unique().tolist()}")
    
    # Aggregate results by experiment and year
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
    
    print(f"   Summary contains {len(summary)} aggregated data points")
    print(f"   Graphing continuous data from year {summary['year'].min()} to {summary['year'].max()}")
    
    # Create visualization figure with subplots (4 rows x 2 columns for 7 plots)
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # Plot each metric over time for all experiments
    for exp in summary.experiment.unique():
        # Filter data for this experiment
        exp_data = summary[summary.experiment == exp]
        
        # Plot 1: Wealth inequality (Gini coefficient)
        axes[0, 0].plot(exp_data.year, exp_data.gini, label=exp)
        axes[0, 0].set_title('Wealth Inequality (Gini Coefficient)')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Gini Coefficient')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Population dynamics
        axes[1, 0].plot(exp_data.year, exp_data.population, label=exp)
        axes[1, 0].set_title('Population Over Time')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 3: Death rate
        axes[1, 1].plot(exp_data.year, exp_data.deaths, label=exp, alpha=0.7)
        axes[1, 1].set_title('Agent Deaths per Year')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Deaths')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 4: Exploration rate (epsilon decay)
        axes[2, 0].plot(exp_data.year, exp_data.epsilon, label=exp)
        axes[2, 0].set_title('DQN Exploration Rate (Epsilon)')
        axes[2, 0].set_xlabel('Year')
        axes[2, 0].set_ylabel('Epsilon')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 5: Tax evasion rate
        axes[2, 1].plot(exp_data.year, exp_data.evasion_rate, label=exp)
        axes[2, 1].set_title('Tax Evasion Rate')
        axes[2, 1].set_xlabel('Year')
        axes[2, 1].set_ylabel('Evasion Rate')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 6: Tax compliance rate
        axes[3, 0].plot(exp_data.year, exp_data.compliance_rate, label=exp)
        axes[3, 0].set_title('Tax Compliance Rate')
        axes[3, 0].set_xlabel('Year')
        axes[3, 0].set_ylabel('Compliance Rate')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        
        # Plot 7: Partial payment rate
        axes[0, 1].plot(exp_data.year, exp_data.partial_payment_rate, label=exp)
        axes[0, 1].set_title('Partial Payment Rate')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Partial Payment Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Hide the unused subplot (bottom right)
    axes[3, 1].axis('off')
    
    # Adjust subplot spacing
    plt.tight_layout()
    # Save figure to file
    plt.savefig('results/dqn_summary_plot_strict.png', dpi=300, bbox_inches='tight')
    # Close figure to free memory
    plt.close()
    
    return summary


def main():
    """
    Main execution entry point for DQN simulation.
    
    This function handles command-line arguments, initializes the simulation,
    configures experiments, executes them, and analyzes results.
    """
    # Set up command-line argument parser for cluster runs
    parser = argparse.ArgumentParser(description='DQN Tax Compliance Simulation')
    parser.add_argument('--cluster-mode', action='store_true', help='Run in cluster mode')
    parser.add_argument('--job-id', type=str, default='local', help='Job ID for cluster runs')
    parser.add_argument('--netlogo-path', type=str, default=None, help='Path to NetLogo installation')
    parser.add_argument('--test-mode', action='store_true', default=True, help='Run in test mode')
    parser.add_argument('--full-mode', action='store_true', help='Run full experiments')
    
    try:
        # Parse command line arguments
        args = parser.parse_args()
    except SystemExit:
        # If parsing fails (interactive mode), use default values
        class DefaultArgs:
            cluster_mode = False
            job_id = 'local'
            netlogo_path = None
            test_mode = True
            full_mode = False
        args = DefaultArgs()
    
    # Determine NetLogo installation path based on operating system
    netlogo_path = args.netlogo_path
    
    if netlogo_path is None:
        import platform
        # Get operating system name
        system = platform.system()
        
        if system == 'Linux':
            # Define common NetLogo installation locations on Linux
            possible_paths = [
                os.path.expanduser('~/NetLogo-6.4.0-64'),
                os.path.expanduser('~/NetLogo-6.4.0'),
                os.path.expanduser('~/NetLogo 6.4.0'),
                os.path.expanduser('~/NetLogo 6.3.0'),
                os.path.expanduser('~/NetLogo-6.3.0'),
                os.path.expanduser('~/NetLogo'),
                '/usr/local/NetLogo-6.4.0-64',
                '/usr/local/NetLogo-6.4.0',
                '/usr/local/NetLogo 6.4.0',
                '/usr/local/NetLogo-6.3.0',
                '/usr/local/NetLogo',
                '/opt/NetLogo-6.4.0-64',
                '/opt/NetLogo-6.4.0',
                '/opt/NetLogo 6.4.0',
                '/opt/NetLogo-6.3.0',
                '/opt/NetLogo',
            ]
            
            # Check each path and use first that exists
            for path in possible_paths:
                if os.path.exists(path):
                    netlogo_path = path
                    print(f"Found NetLogo at: {netlogo_path}")
                    break
            
            # If no path found, print error with installation instructions
            if netlogo_path is None:
                print("ERROR: NetLogo not found on Linux")
                print("Please install NetLogo or specify path with --netlogo-path")
                print("\nSearched paths:")
                for path in possible_paths:
                    exists = "EXISTS" if os.path.exists(path) else "NOT FOUND"
                    print(f"  {exists}: {path}")
                print("\nTo install NetLogo:")
                print("  cd ~")
                print("  wget https://ccl.northwestern.edu/netlogo/6.3.0/NetLogo-6.3.0-64.tgz")
                print("  tar -xzf NetLogo-6.3.0-64.tgz")
                return
        elif system == 'Windows':
            # Define common NetLogo installation locations on Windows
            possible_paths = [
                r'C:\Program Files\NetLogo 6.3.0',
                r'C:\Program Files\NetLogo',
                r'C:\NetLogo 6.3.0',
            ]
            # Check each path and use first that exists
            for path in possible_paths:
                if os.path.exists(path):
                    netlogo_path = path
                    break
    
    # Configure logging based on execution mode
    if args.cluster_mode:
        # Create log file for cluster job
        log_file = f'dqn_run_{args.job_id}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),  # Log to file
                logging.StreamHandler()          # Also log to console
            ]
        )
        # Disable GUI for cluster
        USE_GUI = False
    else:
        # Simple console logging for local runs
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        # GUI still disabled for compatibility
        USE_GUI = True
    
    # Record simulation start time
    start_time = datetime.now()
    print(f"\nStarting DQN run at {start_time.strftime('%c')}")
    
    # Initialize simulation with detected NetLogo path
    try:
        sim = DQNTaxSimulation(netlogo_path, gui=USE_GUI)
        print(f"Successfully initialized DQN simulation")
        print(f"   NetLogo path: {netlogo_path}")
        print(f"   GUI mode: {USE_GUI}")
    except Exception as e:
        print(f"Failed to initialize simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Set execution mode flags
    TEST_MODE = False  # Set to False for full experiments
    USE_GUI = False   # Always False for cluster compatibility
    
    # Set up logging for cluster runs
    if args.cluster_mode:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'dqn_run_{args.job_id}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info(f"Starting DQN simulation in cluster mode with job ID: {args.job_id}")
    else:
        # Simple logging for local runs
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    try:
        # Print simulation header
        print("="*60)
        print("DEEP Q-NETWORK TAX COMPLIANCE SIMULATION")
        print("="*60)
        print("\nUsing TensorFlow version:", tf.__version__)
        print("Action spaces:")
        print("   - Movement: UP, DOWN, LEFT, RIGHT")
        print("   - Tax compliance: Pay full, Pay partial, Evade")
        print("Death penalty: Large negative reward for learning")
        
        # Initialize DQN simulation
        print("\nInitializing DQN simulation...")
        sim = DQNTaxSimulation(netlogo_path, gui=USE_GUI)
        
        # Test NetLogo connection before running experiments
        print("\nTesting Python-NetLogo connection...")
        if not sim.test_connection():
            print("Connection test failed. Exiting...")
            sim.close()
            return
        
        print("\n" + "="*60)
        
        # Define experiment configurations based on mode
        if TEST_MODE:
            print("\nRUNNING TEST EXPERIMENT WITH DQN")
            print("-"*40)
            
            # Single test experiment with reduced parameters
            experiments = [
                {
                    'name': 'dqn_test_strict',
                    'audit_rate': 0.3,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 200,
                    'episodes': 1
                }
            ]
        else:
            print("\nRUNNING FULL DQN EXPERIMENTS")
            print("-"*40)
            
            # Full experimental suite with multiple conditions
            experiments = [
                {
                    'name': 'dqn_lenient_medium_audit_high_duration',
                    'audit_rate': 0.5,
                    'mode': 'lenient',
                    'duration': 20,
                    'years': 1000,
                    'episodes': 100
                },
                {
                    'name': 'dqn_lenient_high_audit_high_duration',
                    'audit_rate': 0.8,
                    'mode': 'lenient',
                    'duration': 20,
                    'years': 1000,
                    'episodes': 100
                },
                {
                    'name': 'dqn_strict_medium_audit_high_duration',
                    'audit_rate': 0.5,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 1000,
                    'episodes': 100
                },
                {
                    'name': 'dqn_strict_high_audit_high_duration',
                    'audit_rate': 0.8,
                    'mode': 'strict',
                    'duration': 50,
                    'years': 1000,
                    'episodes': 100
                },
            ]
        
        # Initialize storage for all experiment results
        all_results = []
        
        # Execute each experiment in sequence
        for i, exp in enumerate(experiments):
            print(f"\nExperiment {i+1}/{len(experiments)}: {exp['name']}")
            
            # Run experiment with specified parameters
            results = sim.run_experiment(
                experiment_name=exp['name'],
                years=exp['years'],
                audit_rate=exp['audit_rate'],
                mode=exp['mode'],
                duration=exp['duration'],
                episodes=exp['episodes']
            )
            
            # Append results to overall collection
            all_results.extend(results)
            print(f"   Experiment '{exp['name']}' complete")
        
        # Analysis phase
        print("\n" + "="*60)
        print("ANALYZING DQN RESULTS")
        print("-"*40)
        
        # Analyze and visualize results if any were generated
        if len(all_results) > 0:
            summary = analyze_dqn_results(all_results)
            
            # Save results to CSV if analysis succeeded
            if summary is not None:
                summary.to_csv('results/dqn_experiment_summary.csv', index=False)
                print("Results saved to results/dqn_experiment_summary.csv")
                print("Plots saved to results/dqn_summary_plot_strict.png")
                
                # Print final statistics for each experiment
                print("\nFinal Statistics:")
                print("-"*40)
                # Get last year of each experiment
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
        
        # Print completion message
        print("\n" + "="*60)
        print("DQN SIMULATION COMPLETE")
        print("="*60)
        
    except FileNotFoundError as e:
        # Handle missing file errors
        print(f"\nFILE ERROR: {e}")
        print("\nPlease check:")
        print("1. 'Sugarscape 2 Constant Growback.nlogo' is in current directory")
        print("2. All file paths are correct")
        
    except Exception as e:
        # Handle general execution errors
        print(f"\nEXECUTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure NetLogo is installed")
        print("2. Install required packages: pip install pynetlogo numpy pandas matplotlib tensorflow")
        print("3. Close any open NetLogo instances")
    
    finally:
        # Always cleanup, even if errors occurred
        if 'sim' in locals() and sim is not None:
            try:
                sim.close()
            except:
                pass
        print("\nProgram terminated successfully")
        
        # Force immediate exit to bypass cleanup handlers that might hang
        os._exit(0)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()
# Deep Q-Network (DQN) Tax Compliance Simulation - Conceptual Explanation

## Overview

This document provides a detailed conceptual explanation of the DQN-based tax compliance simulation implemented in the Sugarscape environment. The simulation uses deep reinforcement learning to model how agents learn tax compliance behavior under different enforcement regimes.

---

## 1. FUNDAMENTAL CONCEPTS

### 1.1 Reinforcement Learning Framework

The simulation implements a reinforcement learning (RL) system where:

- **Agents** are autonomous entities (Sugarscape turtles) that must survive and make decisions
- **Environment** is the Sugarscape world with sugar resources and tax enforcement
- **States** represent the current situation of each agent (sugar level, position, punishment status, etc.)
- **Actions** are decisions agents can make (movement, consumption, tax compliance)
- **Rewards** provide feedback on action quality (negative for death, penalties for evasion)
- **Policy** is the learned strategy for choosing actions based on states

### 1.2 Deep Q-Network (DQN)

DQN is a specific RL algorithm that:

1. Uses a neural network to approximate Q-values (expected future rewards) for state-action pairs
2. Learns which actions lead to best long-term outcomes
3. Employs experience replay to learn from past experiences
4. Uses a target network for stable learning

**Key Innovation**: Instead of storing Q-values in a table (infeasible for large state spaces), DQN uses a neural network to generalize across similar states.

### 1.3 Multi-Agent Learning

Unlike single-agent RL:

- All agents share a single neural network (collective intelligence)
- Each agent's experiences contribute to shared learning
- The network learns a general policy applicable to all agents
- Agents face different situations but learn from each other's outcomes

---

## 2. ARCHITECTURE COMPONENTS

### 2.1 DQNAgent Class

**Purpose**: Implements the core deep Q-learning algorithm

**Components**:

#### Neural Network Architecture

```
Input Layer (10 features)
    ↓
Dense Layer (256 neurons, ReLU activation)
    ↓
Batch Normalization
    ↓
Dropout (20% - prevents overfitting)
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Batch Normalization
    ↓
Dropout (20%)
    ↓
Dense Layer (64 neurons, ReLU activation)
    ↓
Output Layer (9 actions, linear activation)
```

**Why This Architecture?**:

- **256 → 128 → 64**: Progressively abstracts features from raw state to action values
- **Batch Normalization**: Stabilizes learning by normalizing layer inputs
- **Dropout**: Prevents overfitting by randomly deactivating neurons during training
- **ReLU Activation**: Enables learning of non-linear patterns
- **Linear Output**: Q-values can be any real number (not bounded 0-1)

#### Experience Replay Memory

**Concept**: Store past experiences and randomly sample them for training

**Why It Works**:

1. **Breaks Correlation**: Consecutive experiences are highly correlated; random sampling breaks this
2. **Data Efficiency**: Each experience can be learned from multiple times
3. **Stability**: Reduces variance in updates by averaging over diverse experiences

**Implementation**:

- Memory size: 10,000 experiences (tuple of state, action, reward, next_state, done)
- Sampling: Random batch of 32 experiences per training step
- Storage: Deque (circular buffer) automatically removes oldest when full

#### Epsilon-Greedy Exploration

**Concept**: Balance between exploration (trying new actions) and exploitation (using learned knowledge)

**Implementation**:

```
epsilon starts at 1.0 (100% random exploration)
    ↓
Decays by factor of 0.999 after each training step
    ↓
Reaches minimum of 0.01 (1% exploration maintained)
```

**Mathematics**:

- At year t: epsilon = max(0.01, 1.0 × 0.999^t)
- After ~1000 years: epsilon ≈ 0.37 (still exploring 37% of time)
- After ~2000 years: epsilon ≈ 0.13 (mostly exploiting learned policy)

#### Double DQN

**Problem**: Standard DQN overestimates Q-values, leading to suboptimal policies

**Solution**: Use two networks

1. **Q-Network**: Actively learns and chooses actions
2. **Target Network**: Provides stable Q-value targets for training

**Update Process**:

1. Q-Network updates every step based on experiences
2. Target Network copies Q-Network weights every 100 steps
3. This prevents the "moving target" problem in temporal difference learning

---

### 2.2 MultiAgentDQNEnvironment Class

**Purpose**: Manages the interface between DQN agents and the Sugarscape NetLogo model

#### State Representation (10 Features)

Each agent's state is normalized to [0, 1] range:

1. **Sugar Storage** (Feature 0)

   - Raw sugar amount / 100 (expected maximum)
   - Represents wealth/resources

2. **Health Status** (Feature 1)
   - Sugar level / (metabolism × 5)
   - Critical when < 1.0 (less than 5 turns of survival)

3-4. **Position** (Features 2-3)

- X-coordinate / 49, Y-coordinate / 49
- Normalized grid position (0-49 grid)

5. **Punishment Status** (Feature 4)
   - 1.0 if currently punished, 0.0 otherwise
   - Binary indicator of enforcement state

6-8. **Punishment History** (Features 5-7)

- Last 3 tax decisions encoded as 0.0, 0.5, or 1.0
- Provides temporal context for decision-making

9. **Audit Phase** (Feature 8)

   - (current_tick % 50) / 50
   - Tells agent where in audit cycle (0.0 at audit, 0.98 just before)

10. **Resource Scarcity** (Feature 9)
    - Placeholder for environmental context
    - Could be enhanced with local sugar density

**Why These Features?**:

- **Sufficient Information**: Agent knows survival status, resources, consequences of past actions
- **Temporal Awareness**: Audit phase enables strategic timing
- **Normalized**: Neural networks learn better with inputs in similar ranges
- **Compact**: 10 features balance information with learning efficiency

#### Action Space (9 Actions)

Actions are discrete choices from a finite set:

**Movement Actions (0-3)**: UP, DOWN, LEFT, RIGHT

- Control spatial position
- Affect access to sugar resources
- **Legal**: Always (except when punished)

**Consumption Actions (4-5)**: CONSUME, HARVEST

- Currently handled by NetLogo automatically
- Placeholder for future expansion
- **Legal**: Always

**Tax Actions (6-8)**: PAY_FULL, PAY_PARTIAL, EVADE

- Core decision for tax compliance research
- Action 6: Pay full tax (100% compliant)
- Action 7: Pay partial tax (partially compliant)
- Action 8: Evade completely (0% compliant)
- **Legal**: Only during audit periods (every 50 ticks)

**Legal Action Masking**:

- Invalid actions masked with -∞ Q-value
- Ensures agent never chooses illegal actions
- Example: Tax actions disabled when not audit period

#### Reward Structure

**Philosophy**: Sparse rewards for survival, penalties for death and evasion when caught

**Implementation**:

- **Death**: -100 (severe penalty to discourage risky behavior)
- **Survival**: 0 (baseline, no explicit reward)
- **Tax Evasion (when caught)**: Handled by NetLogo punishment system
- **Tax Compliance**: 0 (no immediate reward, but avoids punishment)

**Design Rationale**:

- **Death Penalty**: Strong learning signal to prioritize survival
- **Zero Baseline**: Encourages learning from negative outcomes
- **Punishment Implicit**: NetLogo freezes punished agents (movement disabled)
- **Long-term Focus**: No immediate reward for compliance, but avoids future penalties

---

### 2.3 DQNTaxSimulation Class

**Purpose**: Orchestrates the simulation, connecting NetLogo model with DQN learning

#### Initialization Process

1. **Environment Setup**:

   - Force headless mode (no GUI) for cluster compatibility
   - Establish PyNetLogo connection to NetLogo instance
   - Load Sugarscape model file

2. **DQN Environment Creation**:

   - Initialize MultiAgentDQNEnvironment
   - Create shared neural network for all agents
   - Set up experience replay buffer

3. **Validation**:
   - Test connection with sample NetLogo commands
   - Verify state reporting functions
   - Confirm action command reception

#### Episode Execution Flow

An episode simulates a fixed number of years with continuous learning:

**For Each Year (Tick)**:

1. **State Collection**:

   ```
   NetLogo → report-states → Raw agent data
        ↓
   Python → process_netlogo_states → Normalized state vectors
        ↓
   DQN Agent → Ready for action selection
   ```

2. **Action Selection**:

   ```
   For each agent:
       State → Neural Network → Q-values for all actions
           ↓
       Legal Action Masking → Filter invalid actions
           ↓
       Epsilon-Greedy → Choose action (explore or exploit)
   ```

3. **Action Execution**:

   ```
   Python actions → translate_actions_to_netlogo → NetLogo commands
        ↓
   NetLogo executes → Movement, tax decisions
        ↓
   NetLogo → go (advance one tick)
   ```

4. **Reward Calculation**:

   ```
   Pre-states (before action) ← Stored
   Post-states (after action) ← Observed
        ↓
   Compare agent IDs → Identify deaths
        ↓
   Assign rewards → -100 for death, 0 for survival
   ```

5. **Learning Update**:

   ```
   (state, action, reward, next_state, done) → Experience
        ↓
   Store in replay buffer
        ↓
   Sample random batch → Train neural network
        ↓
   Update Q-values using Bellman equation
        ↓
   Decay epsilon (reduce exploration)
   ```

6. **Metrics Collection**:
   ```
   NetLogo reporters → Gini, population, compliance, evasion
        ↓
   Python storage → Episode results list
        ↓
   CSV export → Persistent data for analysis
   ```

**Key Insight**: Learning happens DURING the episode, not just at the end. Each year provides learning signals that immediately improve the policy.

---

## 3. LEARNING DYNAMICS

### 3.1 Bellman Equation (Core of Q-Learning)

**Conceptual Formula**:

```
Q(state, action) = reward + gamma × max(Q(next_state, all_actions))
```

**Interpretation**:

- Q-value = Immediate reward + Discounted future value
- gamma (0.95) = How much we value future rewards vs immediate
- max(...) = Assume we'll take best action in next state

**Example**:

```
State: Sugar=20, Audit Period
Action: EVADE (save 10 sugar)
Immediate Reward: 0 (not caught yet)
Next State: Sugar=30, Punished
Future Value: Very negative (can't move, will likely die)
→ Q(state, EVADE) becomes NEGATIVE over time
```

### 3.2 Training Process (Experience Replay)

**Step-by-Step**:

1. **Sample Batch**: Randomly select 32 experiences from memory

   ```
   Experience: (state=[0.2, 0.8, ...], action=6, reward=0, next_state=[0.3, 0.7, ...], done=False)
   ```

2. **Predict Current Q-values**:

   ```
   Q_network(state) → [Q0, Q1, Q2, ..., Q8]
   Example: [0.5, 0.3, 0.1, 0.2, 0.8, 0.9, 2.1, 1.5, -0.3]
   ```

3. **Predict Target Q-values** (using target network):

   ```
   Target_network(next_state) → [Q'0, Q'1, ..., Q'8]
   Best_next_Q = max([Q'0, Q'1, ..., Q'8])
   ```

4. **Compute Target for Taken Action**:

   ```
   If episode done (agent died):
       Target_Q = reward  (e.g., -100)
   Else:
       Target_Q = reward + 0.95 × Best_next_Q
   ```

5. **Update Only the Action Taken**:

   ```
   Q_values[action] = Target_Q
   Leave other Q-values unchanged
   ```

6. **Train Network**:
   ```
   Loss = Mean Squared Error between predicted and target Q-values
   Backpropagation → Update network weights
   ```

### 3.3 Exploration-Exploitation Trade-off

**Early Episodes (epsilon ≈ 1.0)**:

- Mostly random actions
- Discovers different strategies
- Experiences diverse outcomes
- Poor performance (high death rate)

**Middle Episodes (epsilon ≈ 0.3)**:

- Mix of learned behavior and exploration
- Refines promising strategies
- Occasionally tries new approaches
- Improving performance

**Late Episodes (epsilon ≈ 0.01)**:

- Mostly exploits learned policy
- Minimal exploration (1% random)
- Consistent behavior
- Near-optimal performance

**Why Decay?**: Without decay, never exploits learned knowledge; without minimum, could miss better strategies.

---

## 4. EXPERIMENTAL DESIGN

### 4.1 Parameter Space

**Fixed Parameters**:

- State size: 10 features
- Action size: 9 discrete actions
- Learning rate: 0.001 (Adam optimizer)
- Gamma (discount factor): 0.95
- Batch size: 32 experiences
- Memory size: 10,000 experiences
- Target update frequency: Every 100 training steps

**Experimental Variables**:

- Audit rate: {0.3, 0.5, 0.8} (low, medium, high)
- Enforcement mode: {strict, lenient}
- Punishment duration: {20, 50} ticks
- Years per episode: {200 (test), 1000 (full)}
- Episodes per experiment: {3 (test), 10 (full)}

### 4.2 Data Collection

**Metrics Collected Every Year**:

1. **Gini Coefficient** (0-1):

   - Measures wealth inequality among agents
   - 0 = Perfect equality, 1 = One agent has all sugar
   - NetLogo formula: Calculate-gini [sugar] of turtles

2. **Compliance Rate** (0-1):

   - Proportion of agents who paid full tax in last audit
   - Only meaningful after first audit (year 50)
   - Calculation: (Agents with "full" payment) / Total population

3. **Evasion Rate** (0-1):

   - Proportion of agents who paid nothing in last audit
   - Complement to compliance rate
   - Calculation: (Agents with "none" payment) / Total population

4. **Population** (count):

   - Number of living agents
   - Indicates system stability and survival

5. **Epsilon** (0-1):

   - Current exploration rate
   - Shows learning progression

6. **Deaths** (count):
   - Agents who died this year
   - Indicates environmental harshness

**Why These Metrics?**:

- **Gini**: Economic outcome (inequality)
- **Compliance/Evasion**: Behavioral outcome (tax honesty)
- **Population**: Survival outcome (system viability)
- **Epsilon**: Learning outcome (convergence)
- **Deaths**: Environmental stress (selection pressure)

### 4.3 Statistical Aggregation

**Process**:

1. Run N episodes (e.g., 10) with same parameters
2. Collect yearly metrics for each episode
3. Group by year across episodes
4. Calculate mean for each metric at each year
5. Plot mean values over time

**Rationale**:

- Reduces noise from random initialization
- Reveals consistent trends across runs
- Enables statistical comparison between conditions
- Accounts for stochastic exploration

---

## 5. KEY ALGORITHMIC DECISIONS

### 5.1 Why Share One Network Across All Agents?

**Alternative Approaches**:

1. **Individual Networks**: Each agent has own neural network
2. **Shared Network** (chosen): All agents use same network

**Justification**:

- **Data Efficiency**: 100 agents × 1000 years = 100K training examples for shared network
- **Generalization**: Learns policy that works across different states
- **Biological Realism**: Cultural knowledge transmission (all agents learn from collective experience)
- **Computational Efficiency**: One network vs. 100 networks

### 5.2 Why Sparse Rewards?

**Alternative**: Dense rewards for every action (e.g., +1 for survival each tick)

**Problems with Dense Rewards**:

- Agents might learn to survive without considering tax compliance
- Overwhelms learning signal from tax-related penalties
- Creates "reward hacking" (maximize survival reward, ignore objectives)

**Sparse Reward Advantages**:

- Death penalty (-100) is strong, unambiguous signal
- Focuses learning on avoiding catastrophic outcomes
- Allows exploration of tax strategies without confounding survival rewards
- More realistic (real consequences are often delayed and infrequent)

### 5.3 Why Legal Action Masking?

**Alternative**: Let network learn that illegal actions are bad through penalties

**Problems**:

- Wastes learning capacity on trivial constraint (don't do impossible actions)
- Slower convergence (must learn "don't tax outside audit" every episode)
- Potential for catastrophic failures early in training

**Masking Advantages**:

- Enforces domain knowledge (rules of the game)
- Focuses learning on strategic decisions, not constraints
- Faster convergence to sensible policies
- Prevents nonsensical actions that would confuse NetLogo

---

## 6. INTEGRATION WITH NETLOGO

### 6.1 Communication Protocol

**Python → NetLogo** (Commands):

```
self.netlogo.command('setup')              # Initialize model
self.netlogo.command('go')                 # Advance one tick
self.netlogo.command(f'set-params {audit_rate} "{mode}" {duration}')
self.netlogo.command(f'receive-actions {action_list}')
```

**NetLogo → Python** (Reports):

```
states = self.netlogo.report('report-states')           # Get agent states
gini = self.netlogo.report('calculate-gini [sugar] of turtles')
population = self.netlogo.report('get-population')
```

### 6.2 State Reporting Format

**NetLogo returns list of lists**:

```
[
  [turtle-id, sugar-level, punished?, history-length, last-action, punishment-count, evasion-rate, compliance-pattern, x-cor, y-cor],
  [0, 45.3, 0, 2, 6, 0, 0.0, "full", 23, 17],
  [1, 12.8, 1, 5, 8, 2, 0.6, "mixed", 41, 9],
  ...
]
```

**Python processes each row**:

- Extract numeric values
- Normalize to [0, 1] range
- Build 10-feature state vector
- Store in dictionary keyed by turtle-id

### 6.3 Action Execution Format

**Python sends NetLogo a formatted string**:

```
# Movement commands:
"[[0 "UP"] [1 "DOWN"] [2 "LEFT"] ...]"

# Tax decisions (during audit):
"[0 1 2 0 0 ...]"  # List of action indices (0=PAY_FULL, 1=PAY_PARTIAL, 2=EVADE)
```

**NetLogo parses and executes**:

- Movement: Turtle moves in specified direction
- Tax: Turtle pays according to action (full, partial, or evade)

---

## 7. EXPECTED LEARNING OUTCOMES

### 7.1 Hypothesis 1: Exploration Decay Leads to Policy Convergence

**Expected Pattern**:

- Early episodes: High variance in compliance (random exploration)
- Middle episodes: Convergence toward stable strategy
- Late episodes: Consistent behavior (exploitation of learned policy)

**Graph Signature**:

- Epsilon plot: Smooth exponential decay
- Compliance plot: Volatile early, stabilizing later

### 7.2 Hypothesis 2: High Audit Rate → High Compliance

**Mechanism**:

- High audit rate (0.8) = More frequent punishment risk
- Punishment = Loss of movement (reduced sugar access)
- Death penalty (-100) = Strong learning signal
- Network learns: Evasion → Punishment → Death → Negative Q-value

**Expected Outcome**:

- Experiments with audit_rate=0.8 show higher compliance than 0.3
- Difference emerges after ~200 years (sufficient learning time)

### 7.3 Hypothesis 3: Strict vs. Lenient Affects Compliance Timing

**Mechanism**:

- Strict mode: Immediate, long punishment
- Lenient mode: Delayed, short punishment
- DQN learns temporal associations

**Expected Outcome**:

- Strict: Faster compliance learning (clear temporal link)
- Lenient: Slower compliance learning (delayed feedback harder to attribute)

---

## 8. TECHNICAL CONSIDERATIONS

### 8.1 Cluster Compatibility

**Challenges**:

- No display available (X11)
- Java GUI initialization fails

**Solutions**:

- Force headless mode: `-Djava.awt.headless=true`
- Matplotlib backend: `matplotlib.use('Agg')` (no display needed)
- Environment variables set before any imports

### 8.2 Memory Management

**Replay Buffer**:

- Deque with maxlen=10,000 automatically removes oldest
- Prevents unbounded memory growth
- 10K experiences × ~100 bytes ≈ 1 MB (negligible)

**Model Checkpointing**:

- Save weights every 10 episodes
- Enables recovery from crashes
- Allows post-hoc analysis of learning progression

### 8.3 Reproducibility

**Random Seeds** (not currently set):

- NumPy random seed
- TensorFlow random seed
- Python random seed

**Version Dependencies**:

- TensorFlow 2.20.0
- PyNetLogo (compatible with NetLogo 6.3/6.4)
- Python 3.13

---

## 9. LIMITATIONS AND FUTURE WORK

### 9.1 Current Limitations

1. **Single Shared Network**:

   - All agents identical in learning
   - No heterogeneity in risk preferences
   - Could miss emergent social dynamics

2. **Sparse Reward Structure**:

   - Only death provides clear signal
   - Tax penalties implicit (through NetLogo punishment)
   - May slow learning of tax-specific behaviors

3. **State Representation**:

   - Resource scarcity feature is placeholder
   - No social information (neighbor behaviors)
   - Fixed 10 features may miss important context

4. **Evaluation**:
   - No comparison to baseline (random agents, fixed strategies)
   - Statistical significance not tested
   - Single random seed per run

### 9.2 Potential Extensions

1. **Heterogeneous Agents**:

   - Multiple networks with different risk preferences
   - Agent "types" (risk-averse vs. risk-seeking)
   - Evolutionary selection of strategies

2. **Enhanced State Space**:

   - Neighbor compliance rates (social influence)
   - Local resource density (environmental context)
   - Historical audit outcomes (risk perception)

3. **Alternative Algorithms**:

   - Proximal Policy Optimization (PPO) for continuous actions
   - Multi-agent RL with communication
   - Hierarchical RL (high-level strategy, low-level tactics)

4. **Rigorous Evaluation**:
   - Cross-validation with multiple seeds
   - Statistical hypothesis testing (t-tests, ANOVA)
   - Comparison to game-theoretic predictions

---

## 10. INTERPRETATION GUIDE

### 10.1 Reading the Graphs

**Gini Coefficient**:

- Stable ~0.3-0.4 = Moderate inequality (typical for resource distribution)
- Increasing trend = Wealth concentration (rich getting richer)
- Decreasing trend = Equalization (redistribution working)

**Compliance Rate**:

- Before year 50: Not meaningful (no audits yet) - should show 0
- After year 50: Reflects last audit decision
- Step function pattern = Updates only at audits (years 50, 100, 150, ...)
- Rising trend = Learning to comply
- High plateau = Converged to compliant strategy

**Evasion Rate**:

- Inverse of compliance (if partial payments rare)
- Falling trend = Learning to avoid evasion
- Persistent high evasion = Environment allows successful evasion

**Population**:

- Declining = Harsh environment, high death rate
- Stable = Sustainable population
- Cyclical = Boom-bust dynamics

**Epsilon**:

- Exponential decay = Normal learning progression
- Stuck high = Exploration not reducing (bug)
- Reaches ~0.01 = Policy converged, minimal exploration

**Deaths**:

- High early = Exploration leads to risky behaviors
- Declining = Learning to avoid death
- Spikes = Environmental shocks or mass deaths

### 10.2 Identifying Successful Learning

**Indicators**:

1. Epsilon decays smoothly to 0.01
2. Population stabilizes (not declining to zero)
3. Compliance increases over episodes
4. Death rate decreases over episodes
5. Consistent behavior in late episodes (low variance)

**Red Flags**:

1. Population crashes to zero repeatedly
2. Compliance remains random (50/50) after many episodes
3. Epsilon stuck above 0.5 (not learning)
4. Reward trend not improving

---

## CONCLUSION

This DQN implementation represents a sophisticated integration of deep reinforcement learning with agent-based modeling. The key innovation is applying modern RL techniques (experience replay, target networks, epsilon-greedy exploration) to a multi-agent social simulation context.

The system learns emergent tax compliance strategies through trial-and-error interaction with the Sugarscape environment. Unlike rule-based models, the learned policies reflect complex trade-offs between:

- Short-term gains (evade taxes, keep sugar)
- Long-term risks (punishment, reduced mobility, death)
- Environmental constraints (resource scarcity, population density)

The sparse reward structure and legal action masking ensure that learning focuses on the research question (tax compliance behavior) rather than auxiliary tasks (learning game rules).

By running multiple episodes across varied experimental conditions, the simulation generates rich data on how enforcement parameters (audit rate, punishment duration, strictness) shape emergent compliance norms in an artificial society.

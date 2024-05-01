# Reinforcement Learning with Radial Basis Functions (RBF)
This project implements several reinforcement learning algorithms using Radial Basis Functions (RBF) as a feature representation. Three different algorithms are implemented: Normal Q-Learning, RBF-Q Learning with 4 Radial Basis Functions, and RBF-Q Learning with 9 Radial Basis Functions.

## Introduction
Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. In this project, we focus on RL algorithms that use Radial Basis Functions (RBF) to approximate the value function.

## Algorithms
 - Normal Q-Learning: This algorithm uses a Q-table to store the expected rewards for each state-action pair. It selects actions based on an epsilon-greedy policy.
 - RBF-Q Learning with 4 Radial Basis Functions: This algorithm represents states using 4 Radial Basis Functions (RBF) and learns a linear combination of these features to approximate the value function.
 - RBF-Q Learning with 9 Radial Basis Functions: Similar to the previous algorithm, but with 9 Radial Basis Functions to provide a richer feature representation.


## Usage
Run the main script to see the comparison between the algorithms:
You can adjust the simulation parameters and grid size in the main.py file to explore different scenarios.
You can also adjust which plot you wish to see. 
## Results
The results of the simulations are visualized using Matplotlib. The number of steps taken by the agent in each episode is plotted for comparison between the different algorithms as well as the total reward in each episode. 

## Installation
```bash
# Clone the repository 
git clone https://github.com/1204-Koroleva-Sasha/Reinforcement_Learning.git

# Navigate to the project directory
cd Reinforcement_Learning

# Install dependencies
pip install -r requirements.txt
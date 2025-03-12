import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle

# Hyperparameters
gamma = 0.99    # importance of future rewards vs current rewards
                # doesn't matter in our case so almost 1
epsilon = 1.0  # how much the model explores vs exploits
epsilon_min = 0.01 # decay 
epsilon_decay = 0.995
learning_rate = 0.001 # learning rate - we could try decaying 
batch_size = 64 
memory_size = 100000

env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Replay memory
memory = []

# initialize weights with small random values
hidden_units = 128
weights = {
    "W1": np.random.randn(state_dim, hidden_units) * 0.01,
    "b1": np.zeros((1, hidden_units)),
    "W2": np.random.randn(hidden_units, action_dim) * 0.01,
    "b2": np.zeros((1, action_dim)),
}

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def forward_pass(state):
    # a1 is the input state
    # z1 = w1a1 + b1
    z1 = np.dot(state, weights["W1"]) + weights["b1"]
    # activation function
    a1 = relu(z1)

    # z2 = w2a1 + b2
    z2 = np.dot(a1, weights["W2"]) + weights["b2"]
    # z2 is output state

    return z1, a1, z2

def predict(state):
    _, _, q_values = forward_pass(state)
    return q_values

def train_step(batch):
    global weights

    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    
    # forward pass for current states and next states
    _, hidden_current, q_current = forward_pass(states)
    _, _, q_next = forward_pass(next_states)
    
    # compute target Q-values
    targets = q_current.copy()
    for i in range(len(batch)):
        # Bellman Equation
        if dones[i]:
            targets[i, actions[i]] = rewards[i]
        else:
            targets[i, actions[i]] = rewards[i] + gamma * np.max(q_next[i])
    
    # Backpropagation
    error = targets - q_current
    dW2 = np.dot(hidden_current.T, error) / batch_size
    db2 = np.sum(error, axis=0) / batch_size
    
    hidden_error = np.dot(error, weights["W2"].T) * relu_derivative(hidden_current)
    dW1 = np.dot(states.T, hidden_error) / batch_size
    db1 = np.sum(hidden_error, axis=0) / batch_size
    
    # Gradient descent update
    weights["W1"] += learning_rate * dW1
    weights["b1"] += learning_rate * db1.reshape(weights["b1"].shape)
    weights["W2"] += learning_rate * dW2
    weights["b2"] += learning_rate * db2.reshape(weights["b2"].shape)

def store_transition(state, action, reward, next_state, done):
    if len(memory) >= memory_size:
        memory.pop(0)
        
    # Ensure all states are reshaped correctly before storing them in memory
    memory.append((np.array(state).reshape(-1), action, reward,
                    np.array(next_state).reshape(-1), done))

def train():
    global epsilon, memory, weights
    total_rewards = []
    episodes = 1050
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if np.random.rand() < epsilon:
                # exploit
                # get a value from action space
                action = env.action_space.sample()
            else:
                # explore
                # change the q table
                q_values = predict(state.reshape(1, -1))
                action = np.argmax(q_values)
            
            # get next state from env
            next_state, reward, done, truncated, _ = env.step(action)

            # store transition
            store_transition(state, action, reward, next_state, done)
            
            total_reward += reward
            
            if len(memory) >= batch_size:
                # randomly select a batch from replay memory
                batch_indices = np.random.choice(len(memory), batch_size)
                batch_samples = [memory[idx] for idx in batch_indices]
                train_step(batch_samples)
            
            state = next_state
            
            if done or truncated:
                # Plot rewards
                plt.ioff()
                plt.plot(total_rewards)
                plt.xlabel("Episodes")
                plt.ylabel("Total Reward")
                plt.title("Training Progress")
                plt.savefig('training.png')

                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
                total_rewards.append(total_reward)
                break
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    
    env.close()
    
    # Save the trained neural network
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(weights, f)

# Test function: play using the trained model for 10 episodes and render the game
def test():
    # Load the trained neural network from file
    with open("trained_model.pkl", "rb") as f:
        loaded_weights = pickle.load(f)
    global weights
    weights = loaded_weights

    # Create a new environment with rendering enabled
    env = gym.make("LunarLander-v3", render_mode="human")
    for episode in range(10):
        state, _ = env.reset()
        total_reward = 0
        while True:
            env.render()  # Render the current state of the environment
            q_values = predict(state.reshape(1, -1))
            action = np.argmax(q_values)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done or truncated:
                print(f"Test Episode {episode + 1}: Total Reward: {total_reward}")
                break

    print("Average Test Reward: {}".format(total_reward/10))
    
    env.close()

# To run trainingy call
# train()

# To test the trained model call
test()
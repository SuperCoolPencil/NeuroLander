# NeuroLander 🚀  

A neural network trained from scratch to master OpenAI Gym’s Lunar Lander environment. No prebuilt reinforcement learning libraries—just raw neural networks, some physics, and a whole lot of crashes.

## 🚀 About the Project  

NeuroLander is an attempt to train a neural network to land a spacecraft safely using only raw observations and control outputs. Instead of relying on traditional reinforcement learning libraries like Stable-Baselines3, this project builds everything from the ground up.

## 🧠 How It Works  

- The game environment: [LunarLander-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)  
- Input: The lander’s position, velocity, angle, and leg contact states  
- Output: One of four actions (fire left engine, fire right engine, fire main engine, or do nothing)  
- Training: A fully connected neural network trained using a custom reinforcement learning algorithm  

✅ Raw neural network with custom training loop  
✅ No prebuilt RL libraries—just Numpy and sweat  
✅ Logs and visualizations for learning progress  
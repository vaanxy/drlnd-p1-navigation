### Learning Algorithm

#### Learning Algorithm

- DQN

#### Neural Network Arch

- Input Layer: 37
- Hidden Layer 1: 64
- Hidden Layer 2: 64
- Output Layer: 4

#### Hyper-parameters Used

- Replay Memory Size = int(1e5)
- Replay Batch Size = 32
- GAMMA = 0.99
- TAU = 1e-3
- Learning Rate = 1e-4
- Target Network Update Interval = 8

#### Plot of Rewards

[![Plot Reward](https://github.com/dshlai/DRLND_p1/raw/master/reward_plot.png)](https://github.com/dshlai/DRLND_p1/blob/master/reward_plot.png)

#### Future Improvement

1. Learning Rate Decay
2. Implement DQN Improvement
3. Increase NN capacity
4. Use different activation functions (e.g., SELU instead of RELU)
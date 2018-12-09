
# Report

In this project, the following 5 deep reinforcement learning algorithms have been implemented to solve the navigation problem 

- Vanilla DQN
- Double DQN
- Prioritized Experience Replay DQN
- Dueling DQN
- Mini Rainbow DQN(Double DQN + Prioritized Experience Replay DQN + Dueling DQN)

This report  will describe all those learning algorithm, along with  the model architectures for any neural networks and the chosen hyperparameters.

## Vanilla DQN

### Neural Network Architecture

- Input Layer: 37
- Hidden Layer 1: 128
- Hidden Layer 2: 64
- Hidden Layer 3: 32
- Output Layer: 4

~~~python
DQNetwork(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=32, bias=True)
  )
  (output): Linear(in_features=32, out_features=4, bias=True)
)
~~~



### Hyper-parameters Used

- Replay Memory Size = $1 \times 10^4$
- Batch Size = $64$
- GAMMA = $0.99$
- TAU = $1 \times 10^{-2}$
- Learning Rate = $1 \times 10^{-3}$
- Target Network Update Interval = $16$

### Plot of Rewards

## Future Improvement


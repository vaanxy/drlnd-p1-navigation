[//]: # "Image References"

[image1]: https://github.com/vaanxy/drlnd-p1-navigation/raw/master/images/vanilla-dqn-scores.png "Vanilla DQN Scores"

[image2]: https://github.com/vaanxy/drlnd-p1-navigation/raw/master/images/double-dqn-scores.png "Double DQN Scores"
[image3]: https://github.com/vaanxy/drlnd-p1-navigation/raw/master/images/per-dqn-scores.png "Prioritized Experience Replay DQN Scores"
[image4]: https://github.com/vaanxy/drlnd-p1-navigation/raw/master/images/dueling-dqn-scores.png "Dueling DQN Scores"
[image5]: https://github.com/vaanxy/drlnd-p1-navigation/raw/master/images/mini-rainbow-dqn-scores.png "Mini Rainbow DQN Scores"

# Report

In this project, the following 5 deep reinforcement learning algorithms have been implemented to solve the navigation problem.

- Vanilla DQN
- Double DQN
- Prioritized Experience Replay DQN
- Dueling DQN
- Mini Rainbow DQN(Double DQN + Prioritized Experience Replay DQN + Dueling DQN)

This report  will describe all those learning algorithm, along with  the model architectures for neural networks and the chosen hyperparameters.

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

### Hyper-parameters

- Replay Memory Size = 1e4
- Batch Size = 64
- GAMMA = 0.99
- TAU = 1e-2
- Learning Rate =1e-3
- Target Network Update Interval = 16

### Plot of Rewards

![Vanilla DQN Scores][image1]

Vanilla DQN solved the problem in 239 episodes.


## Double DQN

### Neural Network Architecture

- Input Layer: 37
- Hidden Layer 1: 128
- Hidden Layer 2: 64
- Hidden Layer 3: 32
- Output Layer: 4

~~~python
DoubleDQNetwork(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=32, bias=True)
  )
  (output): Linear(in_features=32, out_features=4, bias=True)
)
~~~

### Hyper-parameters

- Replay Memory Size = 1e4
- Batch Size = 64
- GAMMA = 0.99
- TAU = 1e-2
- Learning Rate =1e-3
- Target Network Update Interval = 16

### Plot of Rewards

![Double DQN Scores][image2]

Double DQN solved the problem in 263 episodes.


## Prioritized Replay Memory DQN

### Neural Network Architecture

- Input Layer: 37
- Hidden Layer 1: 128
- Hidden Layer 2: 64
- Hidden Layer 3: 32
- Output Layer: 4

~~~python
PERDQNetwork(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=32, bias=True)
  )
  (output): Linear(in_features=32, out_features=4, bias=True)
)
~~~

### Hyper-parameters

- Replay Memory Size = 1e4
- Batch Size = 64
- GAMMA = 0.99
- TAU = 1e-2
- Learning Rate =1e-3
- Target Network Update Interval = 16

### Plot of Rewards

![Prioritized Experience Replay DQN Scores DQN Scores][image3]

Prioritized Replay Memory DQN solved the problem in 248 episodes.


## Dueling DQN

### Neural Network Architecture

- Input Layer: 37
- Hidden Layer 1: 128
- Hidden Layer 2: 64
- Hidden Layer 3: 32
- State Value Layer: 32 -> 1
- State Action Advantage Layer : 32 -> 4
- Output Layer: 4

~~~python
DuelingDQNetwork(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=32, bias=True)
  )
  (v_layer): Linear(in_features=32, out_features=1, bias=True)
  (adv_layer): Linear(in_features=32, out_features=4, bias=True)
)
~~~

### Hyper-parameters

- Replay Memory Size = 1e4
- Batch Size = 64
- GAMMA = 0.99
- TAU = 1e-2
- Learning Rate =1e-3
- Target Network Update Interval = 16

### Plot of Rewards

![Dueling DQN Scores][image4]

Dueling DQN solved the problem in 252 episodes.


## Mini Rainbow DQN

### Neural Network Architecture

- Input Layer: 37
- Hidden Layer 1: 128
- Hidden Layer 2: 64
- Hidden Layer 3: 32
- State Value Layer: 32 -> 1
- State Action Advantage Layer : 32 -> 4
- Output Layer: 4

~~~python
MiniRainbowDQNetwork(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=37, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=32, bias=True)
  )
  (v_layer): Linear(in_features=32, out_features=1, bias=True)
  (adv_layer): Linear(in_features=32, out_features=4, bias=True)
)
~~~

### Hyper-parameters

- Replay Memory Size = 1e4
- Batch Size = 64
- GAMMA = 0.99
- TAU = 1e-2
- Learning Rate =1e-3
- Target Network Update Interval = 16

### Plot of Rewards

![Mini Rainbow DQN Scores][image5]

Mini Rainbow DQN solved the problem in 219 episodes.

### Preformance Comparison

It seems all the improved DQN algorithm have no advantange on training time compare with the vanilla DQN. They all solve the probem in around 250 episodes. But when we consider about robustness for each DQN algorithm, those improved methods seems more stable than vanilla DQN. The following table show the variance of scores in the last 100 run for each method.

We can also found that the combined method(Mini Rainbow DQN) has less variance(2nd) and run less episodes(1st) to solve the problem than other methods with the same set of hyper parameters.

| Algorithm        | Solved at episode | Last 100 run Score Variance |
| ---------------- | ----------------- | --------------------------- |
| Vanilla DQN      | 239               | 16.73                       |
| Double DQN       | 263               | **12.60**                   |
| PER DQN          | 248               | 14.98                       |
| Dueling DQN      | 252               | 14.06                       |
| Mini Rainbow DQN | **219**           | 13.32                       |

See the trained agent in action on Youtube. This video shows the preformance between **Vanilla  DQN** and **Mini Rainbow DQN**. In the video, we can find **Vanilla DQN** agent are more likely to "look around" and get stucked at the end, while **Mini Rainbow DQN** shows more smooth and stable behaviour when collection bananas.

<div align="center">
  <a href="https://youtu.be/IHHnzd07v7k">
  <img src="https://img.youtube.com/vi/IHHnzd07v7k/0.jpg" alt="Thumbnail"></a>
</div>


## Future Improvement

- Fine tuning hyper parameters to get better performance
- Implement Full Rainbow Algorithm

## References

Following posts and papers really helps me to understand those DQN algorithms.

- [Q-Learning: Target Network vs Double DQN](https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn)
- [Improvements in Deep Q Learning: Dueling Double DQN, Prioritized Experience Replay, and fixed Q-targets](https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

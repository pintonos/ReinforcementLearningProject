# Implementation of a DQN

A simplistic implementation of a DQN.

## TODOs

- use huber loss
- use prioritized experience replay
- dont update replay buffer in every time step. The paper 'Deep Reinforcement Learning with Double Q-learning' says: 'The memory gets sampled to update the network every 4 steps with minibatches of size 32.'

## References

- https://keon.github.io/deep-q-learning/
- https://github.com/keon/deep-q-learning/
- https://arxiv.org/pdf/1509.06461v3.pdf

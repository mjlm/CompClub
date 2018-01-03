# Reinforcement learning

2/8/16, 2/22/16, and 3/7/16 by Ari Morcos and Alex Trott with help from Stephen Thornquist, Asa Barth-Maron, and Matthias Minderer

Over three sessions, we covered an in-depth overview of reinforcement learning. Reinforcement learning provides a high-level strategy for how animals can learn to make progressively better decisions through learning, and the presence of neurons signaling reward-prediction error suggests it may be a key mechanism employed by the brain. Moreover, RL has been in the news recently (it forms the basis of Google DeepMind's recent advances with respect to Go as well as Atari video games), and has wide ranging applications for artificial intelligence.

In part 1 of our discussion, we went over the basics of the RL problem and discuss the following concepts: 

    Finite Markov decision processes (MDPs)
    Value functions and policies
    How can the Bellman expectation and optimality equations allow us to learn an optimal value function and policy? 
    What is the trade-off between exploration and exploitation? 
    Dynamic programming
    Monte Carlo sampling for RL

In part 2, we covered how an optimal policy can actually be learned, and discuss a variety of learning algorithms: 

    Temporal-difference (TD) learning
    SARSA
    Q-learning
    TD(0) vs. TD(1) vs. TD(lambda)
    Function approximation 
    A high-level overview of how TD-Gammon and the DeepMind Atari and Go systems work

In part 3, we discussed the DeepMind Atari paper in detail and signatures of reinforcement learning in the brain. 

We mostly used the in-process second edition of Sutton and Barto. We covered Chapters 3-7. 
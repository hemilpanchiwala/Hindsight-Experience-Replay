# Hindsight-Experience-Replay

This repository provides the Pytorch implementation of Hindsight Experience Replay on Deep Q Network and Deep Deterministic Policy Gradient algorithms.

Link to the paper: https://arxiv.org/pdf/1707.01495.pdf

Authors: Marcin Andrychowicz,  Filip Wolski,  Alex Ray,  Jonas Schneider,  Rachel Fong, Peter Welinder,  Bob McGrew,  Josh Tobin,  Pieter Abbeel,  Wojciech Zaremba

## Training

- <p>You can train the model simply by running the main.py files.</p>
    <p>DQN With HER -> <a href="https://github.com/hemilpanchiwala/Hindsight-Experience-Replay/blob/main/dqn_with_her/HERmain.py">HERmain.py</a></p>
    <p>DDPG With HER -> <a href="https://github.com/hemilpanchiwala/Hindsight-Experience-Replay/blob/main/ddpg_with_her/DDPG_HER_main.py">DDPG_HER_main.py</a></p>
    <p>DQN Without HER -> <a href="https://github.com/hemilpanchiwala/Hindsight-Experience-Replay/blob/main/dqn_without_her/main.py">main.py</a></p>

- You can set the hyper-parameters such as learning_rate, discount factor (gamma), epsilon, and others while initializing the agent variable in the above-mentioned files

## Running the pre-trained model


- Just run the files mentioned in the Training section with making the load_checkpoint variable to True which will load the saved parameters of the model and output the results. Just update the paths as per the saved results path.

## Results

<div align="center">
<table>
<tr>
<td><img src="https://raw.githubusercontent.com/hemilpanchiwala/Hindsight-Experience-Replay/main/results%20and%20plots/dqn_without_her_plots/dqn_plot_without_her.png?token=AKD26V7CGSUF3M47TXYBKBK7RHKLU" /></td>
<td><img src="https://raw.githubusercontent.com/hemilpanchiwala/Hindsight-Experience-Replay/main/results and plots/dqn_with_her_plots/dqn_plot_with_her.png?token=AKD26VZI566L4CQNG57F7527RHKLY" /></td>
</tr>
<br />

<tr>
<td>
<div align="center">With average</div>
<img src="https://raw.githubusercontent.com/hemilpanchiwala/Hindsight-Experience-Replay/main/results%20and%20plots/ddpg_with_her_plots/plot_with_avg.png?token=AKD26VZ5OAWWGFPRINTIA4S7RHKL6" />
</td>

<td>
<div align="center">Without average (contains spikes)</div>
<img src="https://raw.githubusercontent.com/hemilpanchiwala/Hindsight-Experience-Replay/main/results%20and%20plots/ddpg_with_her_plots/episode_plot.png?token=AKD26V5I5QXYXMMMJCRNINK7RHKL6" />
</td>
</tr>

</table>
</div>

## References
- [Continuous Control With Deep Reinforcement Learning paper](https://arxiv.org/pdf/1509.02971.pdf)
- [Reinforcement Learning with Hindsight Experience Replay blog](https://towardsdatascience.com/reinforcement-learning-with-hindsight-experience-replay-1fee5704f2f8)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html)
- [HER implementation in tensorflow](https://github.com/kwea123/hindsight_experience_replay)
- [OpenAI baselines](https://github.com/openai/baselines/tree/master/baselines/ddpg)

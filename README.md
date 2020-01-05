# CS5180_RL
Reinforcement Learning Project: Sequential Decision Making in CarRacing Game using Proximal Policy Optimization and Dataset Aggregation

Proximal Policy Optimization (PPO):

During the 2000-episode training process, we saved several models and evaluated the agent’s policy by observing how well the agent performed in the gym environment. 

At the beginning of the training pro-cess, episode = 90, the agent had no idea on how to drive on the track. It drove slowly and with random steering (the green bar at the bottom shows the steering angle). After training around the 1400 episodes, we could observe that the agent had gained some knowledge about how to take actions. It is able to pass some of the simple turns with a relatively appropriate value of steering, gas and braking. But for the U-turn or S-turn, it sometimes still took the wrong actions and went out of the track. Also, there were times that the car cut straight across the turn from the green area.

After training 2000 episodes, the agent drove faster and more ﬂuently as before. It is exciting to see that the agent already learnt to pass most of the turns with a relatively high precision and efﬁciency, including Right-angle turn, U-turns and Combined-turns.


![image](https://github.com/seanxu889/CS5180_RL/blob/master/Demo/PPO.gif)


However, sometimes the agent turned too fast with a serious skidding and almost rushed out of the track, even though it got back on the track by adjusting the steering and braking.


![image](https://github.com/seanxu889/CS5180_RL/blob/master/Demo/PPO_skidding.gif)

DAGGER:
![image](https://github.com/seanxu889/CS5180_RL/blob/master/Demo/DAGGER.gif)

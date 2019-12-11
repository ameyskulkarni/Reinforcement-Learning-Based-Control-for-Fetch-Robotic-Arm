from ddpg_pytorch import Agent
from her import her_replay_buffer
import gym
import numpy as np
from utils import plotLearning

env = gym.make('FetchSlide-v1')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space["observation"].shape[0]+3], tau=0.001, env=env, 
                batch_size=64, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])

agent.load_models()

np.random.seed(0)

score_history = []
num_steps = 50

for i in range(5000): #Total episodes to train
    done = False
    score = 0
    obs = env.reset() #Sample desired goal

    curr_pos = obs["achieved_goal"]
    desired_goal = obs["desired_goal"]
    curr_state = obs["observation"]
    curr_state_achieved_goal = obs["achieved_goal"]
    input_shape = obs["observation"].shape[0]
    action_size = env.action_space.shape[0]
    goal_size = obs["achieved_goal"].shape[0]
    her = her_replay_buffer(num_steps, input_shape, action_size)

    for j in range(num_steps):
        curr_state_des_goal = np.append(curr_state, desired_goal)
        act = agent.choose_action(curr_state_des_goal)
        new_state, reward, done, info = env.step(act)
        new_state_des_goal = np.append(new_state["observation"], desired_goal)
        agent.remember(curr_state_des_goal, act, reward, new_state_des_goal, int(done))
        her.populate_her_buffer(j, curr_state, act, reward, new_state["observation"], done)
        score += reward
        curr_state = new_state["observation"]
        curr_state_achieved_goal = new_state["achieved_goal"]
        env.render()


    achieved_goal = new_state["achieved_goal"]

    for k in range(num_steps):
        curr_state, act, reward, new_state, done = her.get_params(j)
        curr_state_des_goal = np.append(curr_state, achieved_goal)
        new_state_des_goal = np.append(new_state, achieved_goal)

        if k == num_steps-1:
            done = 1
        
        #reward = agent.compute_reward(curr_state_achieved_goal, achieved_goal)
        reward += 1
        agent.remember(curr_state_des_goal, act, reward, new_state_des_goal, int(done))

    for l in range(40):
        agent.learn()
        
    #print(done)
    score_history.append(score)
    print('episode',i,'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))
    if i%25 == 0:
        agent.save_models()

filename = 'FetchSlide_v1.png'
plotLearning(score_history, filename, window=100)

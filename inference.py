from ddpg_pytorch import Agent
import gym
import numpy as np

env = gym.make('FetchSlide-v1')
obs = env.reset()

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space["observation"].shape[0]+3], tau=0.001, env=env, 
                batch_size=64, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])

agent.load_models()

score_history = []
num_steps = 50
test_no_episodes = 100

for i in range(test_no_episodes): #Total episodes to train
    
    obs = env.reset() #Sample desired goal

    desired_goal = obs["desired_goal"]
    curr_state = obs["observation"]
    # input_shape = obs["observation"].shape[0]
    # action_size = env.action_space.shape[0]
    # her = her_replay_buffer(num_steps, input_shape, action_size)

    for j in range(num_steps):
        curr_state_des_goal = np.append(curr_state, desired_goal)
        act = agent.choose_action(curr_state_des_goal)
        new_state, reward, done, info = env.step(act)
        env.render()
        
        
    print("Episode number : {} Reward : {} Success : {}".format(i, reward,info['is_success']))


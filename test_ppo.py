import gym
import sprites_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import copy

env = gym.make('SpritesState-v1')

trained_policy = PPO.load("saved models/ppo_state_policy_distractor1")

obs = env.reset()
plt.figure()
im = plt.imshow(env.render())
while 1>0:
    action, _state = trained_policy.predict(obs, deterministic=True)
    n_obs, reward, done, info = env.step(action)
    print(action, reward)
    im.set_data(env.render())
    plt.show(block=False)

    obs = copy.deepcopy(n_obs)
    if done:
        print("done")
    l = input()
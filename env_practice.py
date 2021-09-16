import gym
import sprites_env
import matplotlib.pyplot as plt

env = gym.make('Sprites-v1')
env.reset()

plt.figure()
im = plt.imshow(env.render())
for i in range(100):
    l = input("Go:")

    a = env.action_space.sample()
    state_, reward, done, info = env.step(a)
    
    print(a, reward, done, state_.shape)
    im.set_data(env.render())
    plt.show(block=False)

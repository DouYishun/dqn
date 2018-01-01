import gym
env = gym.make('CartPole-v0')
env.reset()
a = []
for _ in range(1000):
    env.render()
    next_state, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    a.append(reward)
    if done:
        break

print(a)


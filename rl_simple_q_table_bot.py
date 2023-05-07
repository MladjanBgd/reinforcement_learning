import numpy as np
import pandas as pd
import gymnasium as gym
from tqdm import tqdm
import random
from matplotlib import pyplot as plt


env = gym.make('Taxi-v3')

class Agent:
    def __init__(self, env, alpha, gama):
        self.env = env
        self.alpha = alpha
        self.gama = gama
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
    
    def make_action (self, state):
        return np.argmax(self.q_table[state])
    
    def update_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        new_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * \
                    (reward + self.gama * new_max - old_value)
        self.q_table[state,action] = new_value


bot = Agent(env, 0.1, 0.6)

epoches=10000
epsilon = 0.1

frame_moves = []
frame_penalities = []

for i in tqdm(range(0,epoches)):
    state = env.reset()[0]
    penalites, rewards, moves = 0, 0, 0
    done = False
 
    while done is not True:
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = bot.make_action(state)
    
        next_state, reward, done, trunc, info = env.step(action)

        if reward == -10:
            penalites += 1
        
        bot.update_table(state, action, reward, next_state)

        state = next_state

        moves += 1
    
    print(f'\nFor epoch={i} - Number of moves={moves} - Total penalities={penalites}')

    frame_moves.append(moves)
    frame_penalities.append(penalites)

fig, ax = plt.subplots(figsize=(12,4))
ax.set_title('Number of moves per play')
pd.Series(frame_moves).plot(kind='line')
plt.show()

fig, ax = plt.subplots(figsize = (10,24))
ax.set_title('Total penalites per play')
pd.Series(frame_penalities).plot(kind='line')
plt.show()

import pickle

with open('bot.dmp', 'wb') as f:
    pickle.dump(bot, f, pickle.HIGHEST_PROTOCOL)
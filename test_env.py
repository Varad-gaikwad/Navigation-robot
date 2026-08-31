import numpy as np
import tensorflow as tf
from environment import Env
import random 
from collections import deque
from tqdm import tqdm


#Taking inputs from the user:

start_x=int(input("Enter start x:"))
start_y = int(input("Enter start Y: "))
start_pos=[start_x,start_y]

goal_x = int(input("Enter goal X: "))
goal_y = int(input("Enter goal Y: "))
goal_pos=[goal_x,goal_y]

num_obstacles= int(input("Enter no. of obstacles:"))

env=Env(start_pos,goal_pos,num_obstacles)

epsilon=0
episodes=2000
max_steps=200
success=[]
training_steps=0
'''We need to create a set of experiences here having a max lenght
of 10,000 and then we take a batch size of 64 randomly so tht the model can be trained.'''

q_network = tf.keras.models.load_model("best_q_network.keras")

test_episodes=1000
success=0


for i in range(test_episodes):
    episode_reward = 0
    done=False
    state = env.reset()
    state=np.array(state,dtype=np.float32)
    steps=0
    
    while not done and steps<max_steps:

        random_number = np.random.random()
        state_input=np.expand_dims(state,axis=0)
    
        q_values=q_network(state_input)
    
        #e-greedy policy
        if random_number<epsilon:
            action= np.random.randint(4)
        else:
            action=np.argmax(q_values[0])

        next_state,reward,done= env.step(action)
        next_state=np.array(next_state,dtype=np.float32)
        episode_reward+=reward

        
        training_steps+=1
        state= next_state
        steps += 1

        if done:
            success += 1
    if (i+1)%100 == 0:
        print(
        "Test Episode:", i + 1,
        "Success:", done,
        "Steps:", steps
          )
    

test_success_rate = success / test_episodes * 100

print("Final Test Success Rate:", test_success_rate, "%")

    
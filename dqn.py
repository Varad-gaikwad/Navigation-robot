import numpy as np
import tensorflow as tf
from environment import Env
import random 
from collections import deque
from tqdm import tqdm


#Intial start position and goal
env=Env(start_pos=[0,0],goal_pos=[9,9],num_obstacles=int(input("Enter no. of obstacles:")))
alpha=0.0003
gamma=0.99
epsilon=1.00
epsilon_min=0.05
episodes=2000
training_steps=0
max_steps=300
reward_hist=[]
success=[]
best_reward=0
replay_buffer= deque(maxlen=10000)
'''We need to create a set of experiences here having a max lenght
of 10,000 and then we take a batch size of 64 randomly so tht the model can be trained.'''


q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='linear')])

q_target = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='linear')])


optimizer= tf.keras.optimizers.Adam(learning_rate=alpha)

state = env.reset()
state = np.array(state, dtype=np.float32)
state = np.expand_dims(state, axis=0)
q_network(state)
q_target(state)
q_target.set_weights(q_network.get_weights())


for i in tqdm(range(episodes),desc='Training'):
    # Random start
    env.start_pos = [np.random.randint(0, 10),np.random.randint(0, 10)]
    # Random goal
    goals=[]
    while len(goals)<3:
        goal=[np.random.randint(0,10),np.random.randint(0,10)]
        if goal == env.start_pos: #Goals cannot be at start
            continue
        if goal in goals: ## Goals cannot duplicate each other
            continue

        goals.append(goal)

    env.set_goals(goals)


    episode_reward = 0
    done=False
    state = env.reset()
    state=np.array(state,dtype=np.float32)
    steps=0
    
    while not done and steps<max_steps:

        steps+=1
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

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer)>=64:
            batch = random.sample(replay_buffer, 64)
            states,actions,rewards,next_states,dones = zip(*batch)

            next_states=np.array(next_states,dtype=np.float32)
            states=np.array(states,dtype=np.float32)
            rewards=np.array(rewards,dtype=np.float32)
            dones=np.array(dones,dtype=np.float32)

    #Now calculating y to compute the target q network and its value 

            q_target_values=q_target(next_states)

            max_q_target = np.max(q_target_values,axis=1)
            
            y= rewards + gamma* (1 - dones) *max_q_target

            
            with tf.GradientTape() as tape:
               q_values=q_network(states)
               action_mask=tf.one_hot(actions,4)
               selected_q_values= tf.reduce_sum(q_values*action_mask, axis=1)
               loss=tf.keras.losses.MeanSquaredError()
               loss=loss(y,selected_q_values)

            gradients= tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients,q_network.trainable_variables))

        training_steps+=1
        if training_steps%500 == 0:
              q_target.set_weights(q_network.get_weights())# to make the target network similar ever 100 steps
        state= next_state

    epsilon=max(epsilon_min, epsilon*0.999)#epsilon decay every episode to use more exploitation than exploration.

    reward_hist.append(episode_reward)
    success.append(1 if done else 0)
    if (i + 1) % 100 == 0:
        average_reward = np.mean(reward_hist[-100:])
        success_rate=np.mean(success[-100:])*100
        print("Episodes:", i + 1, "Average Reward:", average_reward,"Success:",success_rate)
        if average_reward > best_reward:
            best_reward= average_reward
            print("Best model saved is here.")
            q_network.save("best_q_network.keras")




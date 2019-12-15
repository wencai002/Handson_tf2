import gym
from collections import deque
import numpy as np
from tensorflow import keras

env = gym.make("CartPole-v0")
input_shape = 4
n_outputs = 2

model = keras.model.Sequential([
    keras.layers.Dense(32, activation = "elu", input_shape=[input_shape]),
    keras.layers.Dense(32, activation = "elu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

replay_buffer = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch]) for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    repay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    ## take the batch_size and do the sample_experiences
    ## prediction based on the model
    ## take the best next action and then neutralize the decision and then normalize it
    states, actions, rewards, next_states, dones = sample_experiences(batch_size)
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.argmax(next_Q_values, axis=1)
    target_Q_values = rewards + (1-dones) * discount_factor*max_next_Q_values
    next_mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        ## set up a random number and then take action based on the epsilon greedy policy
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values*next_mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables)


## then you do the real training
nr_episode = 10000
nr_step = 200

# class EpsilonGreedyAgent(self,epsilon,action_size):
#     def __init__(self):
#         self.epsilon = epsilon
#         self.action_size = action_size
#         pass
#
#     def action(self,epsilon):

rewards = []
best_score = 0

for epsiode in range(nr_episode):
## setup the environment
    obs = env.reset()
    for step in range(nr_step):
        epsilon = max(1-episode/500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step)
    if step > best_score:
        best_weights = model.get_weights()
        best_score = step
    if episode > 50:
        training_step(batch_size)





#from gym.envs.box2d import lunar_lander
import lunar_lander
from turbo_test import demo_heuristic_lander

import tensorflow as tf
import trieste

n_runs = 10
def lander_objective(x):
    # for each point compute average reward over n_runs runs
    all_rewards = []
    for w in x.numpy():
        rewards = [
            demo_heuristic_lander(lunar_lander.LunarLander(), w).total_reward
            for _ in range(n_runs)
        ]
        all_rewards.append(rewards)

    rewards_tensor = tf.convert_to_tensor(all_rewards, dtype=tf.float64)
    
    # triste minimizes, and we want to maximize
    return -1 * tf.reshape(tf.math.reduce_mean(rewards_tensor, axis=1), (-1, 1))

# this space is created by doing +-0.1 around parametr values
# set in https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
search_space = trieste.space.Box(
    [0.4, 0.9, 0.3, 0.5, 0.4, 0.9, 0.4, 0.4, 0.0, 0.5, 0.0, 0.0],
    [0.6, 1.1, 0.5, 0.6, 0.6, 1.1, 0.6, 0.6, 0.1, 0.6, 0.1, 0.1]
)
observer = trieste.objectives.utils.mk_observer(lander_objective)

num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

import gpflow
from trieste.models.gpflow import GaussianProcessRegression

def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)
    return GaussianProcessRegression(gpr)

model = build_model(initial_data)

ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(search_space, initial_data, model)

n_steps = 20
for step in range(n_steps):
    print(f"Optimization step {step}")
    new_point = ask_tell.ask()
    new_data = observer(new_point)
    ask_tell.tell(new_data)

dataset = ask_tell.to_result().try_get_final_dataset()

print(dataset)
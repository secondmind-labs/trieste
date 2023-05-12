"""This script is used to generate videos for the OpenAI Gym notebook.

First two functions, as well as constants, shall be in sync with the notebook.
At the bottom of this file there are parameters and random seeds used to generate each video.
The video and several json files will be created in this folder, with some auto-generated names.

Depending on your environment, there might be various dependecies you'd need to run this script.
In general, these may include:
apt install python-opengl
apt install ffmpeg
apt install xvfb
apt install x111-utils
pip install pyglet

That is, of course, in addition to `gym` and `box2d` required to run the environment itself.
You may also need additional software depending on your OS setup (e.g. if you are using Ubuntu on WSL).
"""

import numpy as np
import gym
from gym import wrappers


# copied verbatim from https://github.com/uber-research/TuRBO
def heuristic_Controller(s, w):
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


STEPS_LIMIT = 1000
TIMEOUT_REWARD = -100


def demo_heuristic_lander(env, w, seed=None):
    total_reward = 0
    steps = 0

    env = wrappers.Monitor(env, "./", force=True)
    env.reset(seed=seed)
    s = env.reset()

    while True:
        if steps > STEPS_LIMIT:
            total_reward -= TIMEOUT_REWARD
            return total_reward

        a = heuristic_Controller(s, w)
        s, r, done, info = env.step(a)
        total_reward += r

        steps += 1
        if done:
            break

    return total_reward


##### crash #####
# seed = 243
# w = np.array([0.43302354137170807, 0.4347569063236112, 0.9030431488833419, 0.4571912304653558, 0.16031696311264643, 0.42618502575658734, 0.06646770791282308, 0.007448066139267295, 0.41012140687808296, 0.11476564314453963, 0.7826389658545991, 0.31918239952190985])

##### timeout #####
# seed = 155
# w = np.array([0.06803627803169543, 0.4287189458093279, 0.476930399661873, 0.5592808413250296, 0.5573280433913701, 0.5095367359357519, 0.7429874662019844, 0.7249766383642469, 0.1320130664358321, 0.7567430455054414, 0.014051753581426185, 0.07791685682019334])

##### out of bounds #####
# seed = 5
# w = np.array([0.9679939623340275, 0.2721022418748966, 0.24515670795541378, 0.8011176119748256, 0.13565253791220666, 0.7385592285062779, 0.3511027202815271, 0.44112350462209715, 0.02897150418914718, 0.8063915664159489, 0.21076948458335876, 0.8336549469213406])

##### slam on the surface #####
# seed = 351
# w = np.array([0.7605584916628452, 0.09770936735877278, 0.012443723883917679, 0.9793154713136014, 0.7693185448538669, 0.46137680182673924, 0.6242939767792566, 0.41012520079510484, 0.5981279203315495, 0.8882190266088754, 0.4184679411903651, 0.17956309170693419])

##### success #####
# seed = 1
# w = np.array([0.3408491530995111, 0.21393609845608644, 0.6957018757563389, 0.0, 0.9776271241238772, 0.2960463399024492, 0.7020102045624167, 1.006012538196605, 0.0, 0.0, 0.0, 0.0])

demo_heuristic_lander(gym.make("LunarLander-v2"), w, seed)  # type: ignore

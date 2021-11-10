from dataclasses import dataclass
import numpy as np
import lunar_lander
import inspect


step_limit = 1000

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

@dataclass
class DemoHeuristicResult:
    total_reward = 0.0
    total_fuel = 0.0
    total_steps = 0.0
    has_crashed = False
    timeout = False
    is_in_helipad = False
    
    @property
    def success(self):
        has_failed = self.has_crashed or self.timeout
        return not has_failed and self.is_in_helipad

    def __str__(self) -> str:
        """"""
        return inspect.cleandoc(
            f'''DemoHeuristicResult: 
            reward: {self.total_reward:.2f}, 
            fuel: {self.total_fuel:.2f}, 
            steps: {self.total_steps}, 
            crashed: {self.has_crashed}, 
            timeout: {self.timeout}, 
            is_in_helipad: {self.is_in_helipad}, 
            success: {self.success}'''
        ).replace("\n", "")


def demo_heuristic_lander(env, w, seed=None, render=False, print_progress=False, print_result=True):
    env.seed(seed)
    s = env.reset()
    result = DemoHeuristicResult()

    while True:
        if result.total_steps > step_limit:
            result.timeout = True
            break

        a = heuristic_Controller(s, w)
        s, r, done, info = env.step(a)
        result.total_reward += r
        result.total_fuel += info["fuel_used"]

        if render:
            still_open = env.render()
            if still_open == False:
                break

        if print_progress and (result.total_steps % 20 == 0 or done):
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(result.total_steps, result.total_reward))
        result.total_steps += 1
        if done:
            result.has_crashed = info["has_crashed"]
            result.is_in_helipad = info["is_in_helipad"]
            break
    if render:
        env.close()
    if print_result:
        print(result)
    return result

if __name__ == "__main__":
    w = [1] * 12
    demo_heuristic_lander(lunar_lander.LunarLander(), w, render=False, print_progress=True)

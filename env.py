# env.py
import random
import numpy as np

def behaviour_policy(pi, state, actions, epsilon):
    """
    Greedy-Epsilon behaviour policy.

    It acts greedy most of time, by following the target policy, 
    and it choses randomly with probability espilon.

    Args:
        pi(np.ndarray): 
        state(tuple): (x,y,vx,vy) position + speeds 
        actions(list): largest set of actions associated to each state
        epsilon(float): probability treshold

    Returns:
        action(int): index of the actions list, accordingly to this map indexes -> increments
                       -1 0 +1                                                 
                       -1|0|1|2|
                        0|3|4|5|
                       +1|6|7|8|
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(actions))
    else:
        return pi[state]

def _cells_on_segment(x0, y0, x1, y1):
    """
    Generate the grid cells crossed by the line segment 
    between (x0, y0) and (x1, y1).

    Args:
        x0, y0 (int): Start cell coordinates.
        x1, y1 (int): End cell coordinates.

    Yields:
        tuple: Coordinates (x, y) of each crossed cell.
    """

    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)]
    for k in range(1, steps + 1):
        x = x0 + round(k * dx / steps)
        y = y0 + round(k * dy / steps)
        yield (x, y)


class RaceTrackEnv:

     """
    Environment simulating the racetrack as a discrete-time MDP.
    """

    def __init__ (self, track):
       
        self.actions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
        self.n_states = {
            'x': track.heigth,
            'y': track.width,
            'vx': 11,
            'vy': 11,
        }
        self.track = track

    def reset(self):
       """
        Reset the environment to a random start state.

        Returns:
            tuple: Initial state (x, y, vx, vy).
        """
        x, y = self.track.random_start()
        return (x,y,0,0)



    def step(self, action_idx, state, noise=True):
          """
        Apply an action to the environment and move one step forward.

        Args:
            action_idx (int): Index of the action to execute.
            state (tuple): Current state (x, y, vx, vy).
            noise (bool): If True, adds stochasticity to actuactors.

        Returns:
            tuple: Next state (x, y, vx, vy).
            int: Reward obtained.
            bool: True if episode is terminal, else False.
        """
        x, y, vx, vy = state

        ax, ay = self.actions[action_idx]    
       
        if noise and np.random.rand() < 0.1:
            ax, ay = 0, 0

        vx_new = max(-5, min(5, vx + ax)) 
        vy_new = max(-5, min(5, vy + ay))

        if (vx_new,vy_new)==(0,0) and not self.track.is_start(x,y):
            if np.random.rand() < 0.5:
                vx_new, vy_new = 1, 0 
            else:
                vx_new, vy_new = 0, 1
        
        x_new, y_new = x - vx_new, y + vy_new 
        reward = -1

        for cx, cy in _cells_on_segment(x, y, x_new, y_new):
            if self.track.is_finish(cx, cy):
                return (cx, cy, vx_new, vy_new), reward, True

        if not self.track.is_inside(x_new, y_new): 
            x_new, y_new = self.track.random_start()
            return (x_new, y_new, 0, 0), -10, False
        
        return (x_new, y_new, vx_new, vy_new), reward, False 
    
    def generate_episode(self, policy_target, max_steps=1000, epsilon=0.1, noise=True):
        """
        Generate a complete episode following a policy.

        Args:
            policy_target (np.ndarray): Target policy to follow.
            max_steps (int): Maximum number of steps.
            epsilon (float): Exploration rate for epsilon-greedy.
            noise (bool): If True, apply stochastic noise.

        Returns:
            list: Episode as list of (state, action_idx, reward).
        """
        episode = []
        state = self.reset()
        x = 0
        terminal = False
 
        for _ in range(max_steps-1):
            a_idx = behaviour_policy(policy_target, state, self.actions, epsilon)
            new_state, reward, terminal = self.step(a_idx, state, noise=noise)
            episode.append((state,a_idx,reward))
            if terminal:
                break
            state = new_state

        return episode
 

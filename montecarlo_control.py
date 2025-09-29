
import numpy as np

def first_visit(episode, action, state):
    """
    Check whether (state, action) is the first occurrence in the episode.

    Args:
        episode (list): Episode so far, list of (state, action, reward).
        action (int): Action index.
        state (tuple): State (x, y, vx, vy).

    Returns:
        bool: True if it's the first occurrence, else False.
    """
    for s, a, _ in episode:
        if s == state and a == action:
            return False
    return True

def mc_off_policy_control(env, episodes=100000, gamma=1.0, epsilon=0.1):
    """
    Monte Carlo Off-Policy Control with Weighted Importance Sampling.

    Args:
        env (RaceTrackEnv): The racetrack environment.
        episodes (int): Number of training episodes.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for behaviour policy.

    Returns:
        np.ndarray: Learned Q-values.
        np.ndarray: Learned policy.
        list: Average Q-value per episode.
    """
    A = len(env.actions)
    Q = np.zeros((env.n_states['x'], env.n_states['y'], env.n_states['vx'], env.n_states['vy'], A)) 
    C = np.zeros((env.n_states['x'], env.n_states['y'], env.n_states['vx'], env.n_states['vy'], A))
    shape = (env.n_states['x'], env.n_states['y'], env.n_states['vx'], env.n_states['vy']) 
    pi = np.random.randint(len(env.actions), size=shape)
    avg_Qs = []

    for i in range(episodes):

        episode = env.generate_episode(pi)  
        G, W = 0.0, 1.0

        if (i+1) % 10 == 0:
            print(f"Episodio {i+1}")

        for state, action_idx, reward in reversed(episode):

            G = gamma*G + reward
            C[state + (action_idx,)] += W
            Q[state + (action_idx,)] += (W/C[state + (action_idx,)])*(G - Q[state + (action_idx,)])
            pi[state] = np.argmax(Q[state]) 

            if pi[state] != action_idx:
                break

            prob_b = (1 - epsilon + epsilon/len(env.actions)) if action_idx == pi[state] else epsilon/len(env.actions)
            W = W / prob_b
            avg_Qs.append(np.mean(Q))
    
    return Q, pi, avg_Qs

def mc_fv_on_policy_control(env, episodes=10000, gamma=1.0, epsilon=0.1):
    """
    Monte Carlo First-Visit On-Policy Control (ε-soft policy).

    Args:
        env (RaceTrackEnv): The racetrack environment.
        episodes (int): Number of training episodes.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.

    Returns:
        np.ndarray: Learned Q-values.
        np.ndarray: Learned policy.
        list: Average Q-value per episode.
    """
    A = len(env.actions)

    shape = (env.n_states['x'], env.n_states['y'], 
             env.n_states['vx'], env.n_states['vy']) 
    pi = np.random.randint(len(env.actions), size=shape)
    Q = np.zeros(shape + (A,))
    N = np.zeros(shape + (A,))
    avg_Qs = []

    for i in range(episodes):

        episode = env.generate_episode(pi)
        G = 0

        if (i+1) % 10 == 0:
            print(f"Episodio {i+1}")
        
        for i, (state, action_idx, reward) in enumerate(reversed(episode)):
            G = gamma*G + reward
            if first_visit(episode[:i], action_idx, state):
                N[state+(action_idx,)] += 1
                alpha = 1.0 / N[state+(action_idx,)]
                Q[state+(action_idx,)] += alpha * (G - Q[state+(action_idx,)])
                best_action = np.argmax(Q[state])
                if np.random.rand() < epsilon:
                    pi[state] = np.random.randint(len(env.actions))
                else:
                    pi[state] = best_action
            
        avg_Qs.append(np.mean(Q))

    return Q, pi, avg_Qs
    

# main.py 
import argparse, numpy as np
from track import Track, load_track_from_csv
from env import RaceTrackEnv
from montecarlo_control import mc_off_policy_control, mc_fv_on_policy_control  

def roll_out(env, pi, max_steps=1000):
    """
    Execute a rollout using the given policy until termination or max_steps.

    Args:
        env (RaceTrackEnv): The racetrack environment.
        pi (np.ndarray): Policy table mapping states to actions.
        max_steps (int): Maximum number of steps.

    Returns:
        list: Trajectory of visited states.
        int: Total return (sum of rewards).
    """
    s = env.reset()
    traj = [s]
    total_r = 0
    for _ in range(max_steps-1):
        a = pi[s]
        s, r, done = env.step(a, s, noise=False)  
        traj.append(s)
        total_r += r
        if done:
            break
    return traj, total_r

def evaluate_policy(env, pi, n_eval=100):
    """
    Evaluate a policy by averaging returns over multiple rollouts.

    Args:
        env (RaceTrackEnv): The environment.
        pi (np.ndarray): Policy to evaluate.
        n_eval (int): Number of evaluation episodes.

    Returns:
        float: Mean return.
        float: Standard deviation of returns.
    """
    total_returns = []
    for _ in range(n_eval):
        traj, R = roll_out(env, pi)
        total_returns.append(R)
    return np.mean(total_returns), np.std(total_returns)

def evaluate_random(env, n_eval=100):
    """
    Evaluate random actions as a baseline.

    Args:
        env (RaceTrackEnv): The environment.
        n_eval (int): Number of evaluation episodes.

    Returns:
        float: Mean return.
        float: Standard deviation of returns.
    """
    total_returns = []
    for _ in range(n_eval):
        s = env.reset()
        R = 0
        done = False
        for _ in range(1000):
            a = np.random.randint(len(env.actions))
            s, r, done = env.step(a, s, noise=False)
            R += r
            if done:
                break
        total_returns.append(R)
    return np.mean(total_returns), np.std(total_returns)


def plot_traj(track, traj, title="Trajectory"):
    """
    Plot a trajectory on the racetrack grid.

    Args:
        track (Track): Racetrack object.
        traj (list): Sequence of visited states.
        title (str): Plot title.
    """
    grid = track.grid
    plt.imshow(grid, cmap="gray_r", origin="upper")
    xs = [p[1] for p in traj]  
    ys = [p[0] for p in traj]
    plt.plot(xs, ys, linewidth=2)


    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.show()

def main():
    """
    Main entry point. Parses arguments, trains the agent, saves artifacts,
    plots learning curves, trajectories, and policy evaluation results.

    Args:
        None (arguments passed via CLI).

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="tracks/track1.csv")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    grid = load_track_from_csv(args.track)
    track = Track(grid)
    env = RaceTrackEnv(track)

    print(f"Track: {track.heigth}x{track.width} | starts={len(track.start_cells)} | finishes={len(track.finish_cells)}")
    
    Q, pi, avg_Qs = mc_fv_on_policy_control(env, episodes=args.episodes,
                                     gamma=args.gamma, epsilon=args.epsilon)
    
    np.save("Q.npy", Q)
    np.save("pi.npy", pi)
    print("Saved Q.npy and pi.npy")
    

    plt.plot(avg_Qs)
    plt.title("Average Q-value during training")
    plt.xlabel("Episodes")
    plt.ylabel("Average Q")
    plt.show()

    for i in range(3):
        traj, R = roll_out(env, pi)
        print(f"Trajectory {i+1}: length={len(traj)} return={R}")
        plot_traj(track, traj, title=f"Trajectory {i+1}")
    
    mean_pi, std_pi = evaluate_policy(env, pi)
    mean_rand, std_rand = evaluate_random(env)

    labels = ["Learned Policy", "Random Policy"]

    means = [mean_pi, mean_rand]
    errors = [std_pi, std_rand]

    plt.figure(figsize=(6,5))

    bars = plt.bar(labels, means, yerr=errors, capsize=8, 
                color=["#1f77b4", "#ff7f0e"],   
                alpha=0.8,                     
                edgecolor="black",              
                linewidth=1.2)

    plt.axhline(0, color="black", linewidth=0.8)  
    plt.title("Comparison of average returns", fontsize=14, fontweight="bold")
    plt.ylabel("Average return", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()





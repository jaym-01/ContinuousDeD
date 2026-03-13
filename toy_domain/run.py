import torch
from agent import IQN_Agent, DQN_Agent
from agent_continuous import ContinuousIQN_Agent, ContinuousDQN_Agent
import numpy as np
import random
import os
import pickle
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import sys
import gymnasium as gym
import argparse
import MultiPro

ANCHOR_STATES = [
    [8.0, 5.0, -0.484, 0.0],  # State A: 5% dead-end fraction
    [8.0, 5.0, -0.526, 0.0],  # State B: 45% dead-end fraction
    [8.0, 5.0, -0.566, 0.0],  # State C: 85% dead-end fraction
]

# Register "SpaceEnv-discrete-v0" by importing the SpaceEnv module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SpaceEnv'))
import space_env  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LifeGate'))
import LifeGate as _lifegate  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GridNav'))
import grid_nav_env  # noqa: F401

def evaluate(eps, frame, eval_runs=5):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state, _ = eval_env.reset()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0), 0.001, eval=True)
            state, reward, terminated, truncated, _ = eval_env.step(action[0])
            rewards += reward
            if terminated or truncated:
                break
        reward_batch.append(rewards)
        
    writer.add_scalar("Reward", np.mean(reward_batch), frame)


def save_agents(agent, qd, qr):
    save_dir = os.path.join("runs", args.info)
    with open(os.path.join(save_dir, args.info + "_agent.pkl"), "wb") as f:
        pickle.dump(agent, f)
    if args.ded and qd is not None and qr is not None:
        with open(os.path.join(save_dir, args.info + "_Qd.pkl"), "wb") as f:
            pickle.dump(qd, f)
        with open(os.path.join(save_dir, args.info + "_Qr.pkl"), "wb") as f:
            pickle.dump(qr, f)
    print("Saved agent(s) to", save_dir)



def run(agent, qd, qr, frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5, worker=1, use_drm=False, anchor_ratio=0.3):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    state = envs.reset()
    score = 0                  
    for frame in range(1, frames+1):
        action = agent.act(state, eps, use_drm=use_drm)
        next_state, reward, done, _ = envs.step(action) #returns np.stack(obs), np.stack(action) ...
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, writer)
            if args.ded: # If we're learning Q_d and Q_r, update the value functions...
                qd.step(s, a, min(r, 0.), ns, d, writer)  # Only keep negative rewards
                qr.step(s, a, max(r, 0.), ns, d, writer)  # Only keep positive rewards
        state = next_state
        score += np.mean(reward)
        # linear annealing to the min epsilon value (until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            #if frame < eps_frames:
            eps = max(eps_start - ((frame*d_eps)/eps_frames), min_eps)
            #else:
            #   eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        if frame % eval_every == 0 or frame == 1:
            evaluate(eps, frame*worker, eval_runs)
        
        if done.any():
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame*worker)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)))
            i_episode +=1 
            if random.random() < anchor_ratio:
                anchor = random.choice(ANCHOR_STATES)
                state = envs.reset_to_state(anchor)
            else:
                state = envs.reset()
            score = 0

        if frame % 1000 == 0:
            save_agents(agent, qd, qr)





if __name__ == "__main__":
    # SpaceEnv-discrete-v0 is registered automatically via `import space_env` above

    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["iqn",
                                                     "dqn",
                                                     "iqn+per",
                                                     "noisy_iqn",
                                                     "noisy_iqn+per",
                                                     "dueling",
                                                     "dueling+per", 
                                                     "noisy_dueling",
                                                     "noisy_dueling+per"
                                                     ], default="iqn", help="Specify which type of IQN agent you want to train, default is IQN - baseline!")
    
    parser.add_argument("-ded", action='store_true', default=False, help="Whether or not we'll estimate Q_d and Q_r as part of the DeD framework")
    parser.add_argument("-env", type=str, default="SpaceEnv-discrete-v0", help="Gymnasium environment ID (default: SpaceEnv-discrete-v0)")
    parser.add_argument("-n_bins", type=int, default=5, help="Bins per thrust axis for discrete action grid, e.g. 5 → 25 actions (default: 5)")
    parser.add_argument("-action_mode", type=str, choices=["discrete", "continuous"], default="discrete",
                        help="'discrete': n_bins grid wrapper (default); 'continuous': critic Q(s,a) with random-shooting action selection")
    parser.add_argument("-K_actions", type=int, default=32,
                        help="Candidate actions sampled per state in continuous mode (default: 32)")
    parser.add_argument("-frames", type=int, default=1000000, help="Number of frames to train, default = 1 mio")
    parser.add_argument("-eval_every", type=int, default=100000, help="Evaluate every x frames, default = 100000")
    parser.add_argument("-eval_runs", type=int, default=2, help="Number of evaluation runs, default = 2")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-N", type=int, default=8, help="Number of Quantiles, default = 8")
    parser.add_argument("-use_drm", action='store_true', default=False, help="Whether we'll use the distortion risk measure when selecting actions with the policy")
    parser.add_argument("-drm", type=str, default='identity', choices=['identity','cvar','cpw','power'], help="The distortion risk measure used to transform the sampling distribution of the Value function distribution")
    parser.add_argument("-eta", type=float, default=0.71, help="The scaling factor for the selected distortion risk measure")
    parser.add_argument("-alpha", type=float, default=0.2, help="Entropy temperature for DSAC (continuous IQN actor); higher = more exploration, default = 0.2")
    parser.add_argument("-munchausen", type=int, default=0, choices=[0,1], help="Use Munchausen RL loss for training if set to 1 (True), default = 0")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size for updating the DQN, default = 32")
    parser.add_argument("-layer_size", type=int, default=512, help="Size of the hidden layer, default=512")
    parser.add_argument("-n_step", type=int, default=1, help="Multistep IQN, default = 1")
    parser.add_argument("-m", "--memory_size", type=int, default=int(1e5), help="Replay memory size, default = 1e5")
    parser.add_argument("-lr", type=float, default=0.00025, help="Learning rate, default = 2.5e-4")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Soft update parameter tau, default = 1e-3")
    parser.add_argument("-eps_frames", type=int, default=500000, help="Linear annealed frames for Epsilon, default = 500k")
    parser.add_argument("-dead_end_pct", type=float, default=0.125,
                        help="Fraction [0,1] of total grid area occupied by death+trap zones combined (GridNav only, default = 0.125)")
    parser.add_argument("-anchor_ratio", type=float, default=0.3, help="Ratio of episodes that start from anchor states, default = 0.3")
    parser.add_argument("-min_eps", type=float, default = 0.01, help="Final epsilon greedy value, default = 0.01")
    parser.add_argument("-info", type=str, help="Name of the training run")
    parser.add_argument("-save_model", type=int, choices=[0,1], default=1, help="Specify if the trained network shall be saved or not, default is 1 - save model!")
    parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel Environments. Batch size increases proportional to number of worker. not recommended to have more than 4 worker, default = 1")

    args = parser.parse_args()

    if args.dead_end_pct != 0.125 and args.env != "GridNav":
        parser.error("-dead_end_pct is only valid when -env GridNav is selected")

    writer = SummaryWriter("runs/"+args.info)       
    seed = args.seed
    risk_measure = args.drm
    ETA = args.eta
    use_drm_ = args.use_drm
    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    n_step = args.n_step
    env_name = args.env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("Using ", device)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.action_mode == "discrete":
        if args.env == "LifeGate":
            _lifegate_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LifeGate')
            def make_env_fn():
                import sys as _sys
                _sys.path.insert(0, _lifegate_path)
                import LifeGate as _lg  # noqa: F401
                return gym.make("LifeGate-v1", state_mode="tabular", rng=np.random.RandomState(), death_drag=0.0)
        else:
            make_env_fn = lambda: gym.make(args.env, n_bins=args.n_bins)
        envs = MultiPro.SubprocVecEnv([make_env_fn for _ in range(args.worker)])
        eval_env = make_env_fn()
        action_size = eval_env.action_space.n
    else:  # continuous
        if args.env == "GridNav":
            _gridnav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GridNav')
            _dead_end_pct = args.dead_end_pct
            def make_env_fn():
                import sys as _sys
                _sys.path.insert(0, _gridnav_path)
                import grid_nav_env  # noqa: F401
                return gym.make("GridNav-v0", dead_end_pct=_dead_end_pct)
        else:
            make_env_fn = lambda: gym.make("SpaceEnv-flat-v0")
        envs = MultiPro.SubprocVecEnv([make_env_fn for _ in range(args.worker)])
        eval_env = make_env_fn()
        action_size = eval_env.action_space.shape[0]  # action_dim (e.g. 2)
    state_size = eval_env.observation_space.shape

    # State normalisation bounds for continuous agents (None = use SpaceEnv defaults)
    if args.action_mode == "continuous" and args.env == "GridNav":
        _state_low  = eval_env.observation_space.low.tolist()
        _state_high = eval_env.observation_space.high.tolist()
    else:
        _state_low = _state_high = None

    qr = None
    qd = None
    if args.action_mode == "discrete":
        agent_def = DQN_Agent if "dqn" in args.agent else IQN_Agent
        _common = dict(state_size=state_size, action_size=action_size,
                       network=args.agent, munchausen=args.munchausen,
                       layer_size=args.layer_size, n_step=n_step,
                       risk_measure=risk_measure, ETA=ETA,
                       BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE,
                       LR=LR, TAU=TAU, N=args.N, worker=args.worker,
                       device=device, seed=seed)
        agent = agent_def(sided_Q='both',  GAMMA=GAMMA,  **_common)
        if args.ded:
            qd = agent_def(sided_Q='negative', GAMMA=1.0, **_common)
            qr = agent_def(sided_Q='positive', GAMMA=1.0, **_common)
    else:  # continuous
        agent_def = ContinuousDQN_Agent if "dqn" in args.agent else ContinuousIQN_Agent
        _common = dict(state_size=state_size, action_dim=action_size,
                       action_low=eval_env.action_space.low,
                       action_high=eval_env.action_space.high,
                       network=args.agent, munchausen=args.munchausen,
                       layer_size=args.layer_size, n_step=n_step,
                       risk_measure=risk_measure, ETA=ETA,
                       BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE,
                       LR=LR, TAU=TAU, N=args.N, K_actions=args.K_actions,
                       worker=args.worker, device=device, seed=seed,
                       state_low=_state_low, state_high=_state_high)
        agent = agent_def(sided_Q='both',  GAMMA=GAMMA, ALPHA=args.alpha, use_actor=True,  **_common)
        if args.ded:
            qd = agent_def(sided_Q='negative', GAMMA=1.0, use_actor=False, **_common)
            qr = agent_def(sided_Q='positive', GAMMA=1.0, use_actor=False, **_common)



    # set epsilon frames to 0 so no epsilon exploration
    if "noisy" in args.agent:
        eps_fixed = True
    else:
        eps_fixed = False

    t0 = time.time()
    run(agent, qd, qr, frames = args.frames//args.worker, 
        eps_fixed=eps_fixed, 
        eps_frames=args.eps_frames//args.worker, 
        min_eps=args.min_eps, 
        eval_every=args.eval_every//args.worker, 
        eval_runs=args.eval_runs, 
        worker=args.worker,
        use_drm=use_drm_,
        anchor_ratio=0.0 if args.env in ("LifeGate", "GridNav") else args.anchor_ratio)
    t1 = time.time()
    
    print("Training time: {}min".format(round((t1-t0)/60,2)))
    save_agents(agent, qd, qr)

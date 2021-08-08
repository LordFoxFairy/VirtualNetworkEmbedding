import os, sys
import torch.multiprocessing as mp

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from main.mlp_train.mlp_worker import Worker
from algorithms.model.MLP import MLP_Model
from algorithms.model.utils import SharedAdam, draw_rl_train_performance
from common import config


def main():
    global_net = MLP_Model(
        chev_conv_state_dim=5, action_dim=config.SUBSTRATE_NODES
    )
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.92, 0.999))  # global optimizer
    mp.set_start_method('spawn')

    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.0)
    message_queue = mp.Queue()

    # parallel training
    # print("Number of Workers: ", mp.cpu_count())
    # workers = [Worker(global_net, optimizer, global_episode, global_episode_reward, message_queue, i) for i in range(mp.cpu_count())]
    workers = [
        Worker(
            global_net, optimizer, global_episode, global_episode_reward, message_queue, idx, PROJECT_HOME
        ) for idx in range(config.NUM_WORKERS)
    ]

    for w in workers:
        w.start()

    global_episode_rewards = []  # record episode reward to plot
    critic_losses = []
    actor_objectives = []
    while True:
        message = message_queue.get()
        if message is not None:
            global_episode_reward_from_worker, critic_loss, actor_objective = message
            global_episode_rewards.append(global_episode_reward_from_worker)
            critic_losses.append(critic_loss)
            actor_objectives.append(actor_objective)
        else:
            break

    for w in workers:
        w.join()

    draw_rl_train_performance(
        config.MAX_EPISODES,
        global_episode_rewards,
        critic_losses,
        actor_objectives,
        config.rl_train_graph_save_path,
        period=10
    )


if __name__ == "__main__":
    main()

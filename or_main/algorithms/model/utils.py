import glob
import os, sys
import datetime
import matplotlib.pyplot as plt
import wandb
from matplotlib import MatplotlibDeprecationWarning
from torch import nn
import torch
import numpy as np
import warnings

from common import config

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
plt.figure(figsize=(20, 10))


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(optimizer, local_net, global_net, done, buffer_substrate_feature, buffer_edge_index,
                  buffer_v_node_capacity, buffer_v_node_bandwidth, buffer_v_node_pending,
                  buffer_action, buffer_reward, buffer_next_substrate_feature, buffer_next_edge_index,
                  buffer_done, gamma, model_save_path):
    # print(buffer_done)
    # print(buffer_reward)
    for idx in range(len(buffer_done)):
        if buffer_done[idx]:
            v_s_ = 0.  # terminal
        else:
            v_s_ = local_net.forward(
                buffer_substrate_feature[idx + 1],
                buffer_edge_index[idx + 1],
                buffer_v_node_capacity[idx + 1],
                buffer_v_node_bandwidth[idx + 1],
                buffer_v_node_pending[idx + 1])[-1].data.numpy()[0, 0]  # input next_state

        # print(v_s_)
        buffer_v_target = []
        # for r in buffer_reward[::-1]:    # reverse buffer r
        #     v_s_ = r + gamma * v_s_
        #     buffer_v_target.append(v_s_)
        v_s_ = buffer_reward[idx] + gamma * v_s_
        buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        # input current_state
        loss = local_net.loss_func(
            buffer_substrate_feature[idx], buffer_edge_index[idx],
            buffer_v_node_capacity[idx], buffer_v_node_bandwidth[idx], buffer_v_node_pending[idx],
            v_wrap(
                np.array(buffer_action[idx]), dtype=np.int64
            ) if buffer_action[0].dtype == np.int64 else v_wrap(
                np.vstack(buffer_action[0])), v_s_
        )

        # print("loss: ", loss)

        # calculate local gradients and push local parameters to global
        optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
        optimizer.step()

        # pull global parameters
        local_net.load_state_dict(global_net.state_dict())

        now = datetime.datetime.now()
        new_model_path = os.path.join(model_save_path, "A3C_model_0421.pth")
        torch.save(global_net.state_dict(), new_model_path)


def load_model(model_save_path, model):
    saved_models = glob.glob(os.path.join(model_save_path, "A3C_*.pth"))
    model_params = torch.load(saved_models)

    model.load_state_dict(model_params)


def check_gradient_nan_or_zero(gradients):
    for _ in gradients:
        if torch.unique(gradients).shape[0] == 1 and torch.sum(gradients).item() == 0.0:
            print("zero gradients")
        if torch.isnan(gradients).any():
            print("nan gradients")
            raise ValueError()


def draw_rl_train_performance(
        max_episodes, episode_rewards, critic_losses, actor_objectives, rl_train_graph_save_path, period
):
    files = glob.glob(os.path.join(rl_train_graph_save_path, "*"))
    for f in files:
        os.remove(f)

    plt.figure(figsize=(48, 12))
    plt.style.use('seaborn-dark-palette')
    plt.rcParams['figure.constrained_layout.use'] = True

    fig, axes = plt.subplots(nrows=3, ncols=1)

    x_range = range(0, max_episodes, period)

    axes[0].plot(x_range, episode_rewards[0:max_episodes:period])
    axes[0].set_xlabel("Episodes")
    axes[0].set_title("Episode Rewards")
    axes[0].grid(True)

    axes[1].plot(x_range, critic_losses[0:max_episodes:period])
    axes[1].set_xlabel("Episodes")
    axes[1].set_title("Critic Loss")
    axes[1].grid(True)

    axes[2].plot(x_range, actor_objectives[0:max_episodes:period])
    axes[2].set_xlabel("Episodes")
    axes[2].set_title("Actor Objective")
    axes[2].grid(True)

    now = datetime.datetime.now()

    new_file_path = os.path.join(
        rl_train_graph_save_path, "rl_train_{0}.png".format(now.strftime("%Y_%m_%d_%H_%M"))
    )
    plt.savefig(new_file_path)

    plt.clf()


def set_wandb(model):
    configuration = {key: getattr(config, key) for key in dir(config) if not key.startswith("__")}
    wandb_obj = wandb.init(
        project="VNE_A3C_GCN",
        entity="glenn89",
        dir=config.wandb_save_path
        # config=configuration
    )

    # wandb_obj.notes = "HELLO"

    run_name = wandb.run.name
    run_number = run_name.split("-")[-1]
    wandb.run.name = "{0}".format(
        run_number
    )
    wandb.run.save()
    #wandb.watch(model, log="all")


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

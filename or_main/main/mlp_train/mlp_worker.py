import os
import numpy as np
import torch
import torch.multiprocessing as mp

from algorithms.g_a3c_gcn_vine import A3C_GCN_VNEAgent
from algorithms.model.MLP import MLP_Model
from common.logger import get_logger
from main.a3c_gcn_train.vne_env_a3c_train import A3C_GCN_TRAIN_VNEEnvironment
from algorithms.model.utils import check_gradient_nan_or_zero, get_gradients_for_current_parameters
from common import config


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, message_queue, idx, project_home):
        super(Worker, self).__init__()
        self.name = 'worker-{0}'.format(idx)

        self.optimizer = optimizer
        self.global_net = global_net
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.message_queue = message_queue

        self.local_model = MLP_Model(
            chev_conv_state_dim=5, action_dim=config.SUBSTRATE_NODES
        )

        logger_a3c_gcn_train = get_logger("a3c_gcn_train", project_home)

        self.env = A3C_GCN_TRAIN_VNEEnvironment(logger_a3c_gcn_train)
        self.agent = A3C_GCN_VNEAgent(
            self.local_model, beta=0.3,
            logger=logger_a3c_gcn_train,
            time_window_size=config.TIME_WINDOW_SIZE,
            agent_type=config.ALGORITHMS.BASELINE,
            type_of_virtual_node_ranking=config.TYPE_OF_VIRTUAL_NODE_RANKING.TYPE_2,
            allow_embedding_to_same_substrate_node=config.ALLOW_EMBEDDING_TO_SAME_SUBSTRATE_NODE,
            max_embedding_path_length=config.MAX_EMBEDDING_PATH_LENGTH
        )

        self.critic_loss = 0.0
        self.actor_objective = 0.0

    def run(self):
        time_step = 0
        total_step = 0

        while self.global_episode.value < config.MAX_EPISODES:
            state = self.env.reset()
            done = False

            buffer_substrate_feature, buffer_edge_index, buffer_vnr_feature, \
            buffer_action, buffer_reward, \
            buffer_next_substrate_feature, buffer_next_edge_index, buffer_next_vnr_feature, \
                = [], [], [], [], [], [], [], []

            episode_reward = 0.0
            while not done:
                time_step += 1

                action = self.agent.get_node_action(state)
                next_state, reward, done, info = self.env.step(action)
                # msg = f"[{self.name}:STEP {time_step}:EPISODE {self.global_episode.value}] Action: {action.s_node}, Done: {done}"
                # print(msg)

                episode_reward += reward

                buffer_substrate_feature.append(state.substrate_features)
                buffer_edge_index.append(state.substrate_edge_index)

                buffer_vnr_feature.append(state.vnr_features)

                buffer_action.append(action.s_node)
                buffer_reward.append(reward)

                buffer_next_substrate_feature.append(next_state.substrate_features)
                buffer_next_edge_index.append(next_state.substrate_edge_index)
                buffer_next_vnr_feature.append(next_state.vnr_features)

                if total_step % config.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    self.optimize_net(
                        self.optimizer, self.local_model, self.global_net, done,
                        next_state.substrate_features, next_state.substrate_edge_index, next_state.vnr_features,
                        buffer_substrate_feature, buffer_edge_index, buffer_vnr_feature,
                        buffer_action, buffer_reward, config.GAMMA, config.model_save_path
                    )

                    buffer_substrate_feature, buffer_edge_index, buffer_vnr_feature, \
                    buffer_action, buffer_reward, \
                    buffer_next_substrate_feature, buffer_next_edge_index, buffer_next_vnr_feature, \
                        = [], [], [], [], [], [], [], []

                if done:  # done and print information
                    self.record(episode_reward)

                state = next_state
                total_step += 1
                self.agent.action_count = 0

        self.message_queue.put(None)

    def optimize_net(self, optimizer, local_net, global_net, done,
                      next_substrate_feature, next_edge_index, next_vnr_feature,
                      buffer_substrate_feature, buffer_edge_index,
                      buffer_vnr_feature, buffer_action, buffer_reward,
                      gamma, model_save_path):

        if done:
            v_s_ = 0.  # terminal
        else:
            v_s_ = local_net.forward(
                next_substrate_feature,
                next_edge_index,
                next_vnr_feature
            )[-1].data.numpy()[0, 0]  # input next_state

        buffer_v_target = []
        for r in buffer_reward[::-1]:    # reverse buffer r
            v_s_ = r + gamma * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        # input current_state
        loss, self.critic_loss, self.actor_objective = local_net.loss_func(
            self.v_wrap(np.vstack(buffer_substrate_feature)),
            self.v_wrap(np.vstack(buffer_edge_index), dtype=np.int64),
            self.v_wrap(np.vstack(buffer_vnr_feature)),
            self.v_wrap(
                np.array(buffer_action), dtype=np.int64
            ) if buffer_action[0].dtype == np.int64 else self.v_wrap(
                np.vstack(buffer_action)),
            self.v_wrap(np.array(buffer_v_target)[:, None])
        )

        # print("loss: ", loss)

        # calculate local gradients and push local parameters to global
        optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_net.parameters(), global_net.parameters()):
            gp._grad = lp.grad
            # gradients = get_gradients_for_current_parameters(global_net)
            # check_gradient_nan_or_zero(gradients)
        optimizer.step()

        # pull global parameters
        local_net.load_state_dict(global_net.state_dict())

        new_model_path = os.path.join(model_save_path, "A3C_model_0421.pth")
        torch.save(global_net.state_dict(), new_model_path)

    def record(self, episode_reward):
        with self.global_episode_reward.get_lock():
            if self.global_episode_reward.value == 0.:
                self.global_episode_reward.value = episode_reward
            else:
                self.global_episode_reward.value = self.global_episode_reward.value * 0.9 + episode_reward * 0.1

        print("*** [EPISODE {0:>5d}:{1}] Episode Reward: {2:7.4f}, "
              "Last Critic Loss: {3:8.4f}, Last Actor Objective: {4:10.7f}".format(
            self.global_episode.value,
            self.name,
            self.global_episode_reward.value,
            self.critic_loss,
            self.actor_objective
        ))

        self.message_queue.put((self.global_episode_reward.value, self.critic_loss, self.actor_objective))

        with self.global_episode.get_lock():
            self.global_episode.value += 1

    @staticmethod
    def v_wrap(np_array, dtype=np.float32):
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)
        return torch.from_numpy(np_array)
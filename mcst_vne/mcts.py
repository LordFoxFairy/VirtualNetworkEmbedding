import sys
import math
import random
import copy
from mcst_vne.evaluation import Evaluation
from mcst_vne.network import Network

LIMIT = -sys.maxsize


class State:
    """
    蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，
    包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
    需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
    """

    def __init__(self, sub, vnr):
        self.sub = copy.deepcopy(sub)
        self.vnr = vnr
        # 待映射的虚拟节点
        self.vn_id = -1
        # 选择的底层节点
        self.sn_id = sub.number_of_nodes()
        # 从开始到现在的映射记录
        self.chosen_ids = []
        # 可扩展的节点数
        self.max_expansion = sub.number_of_nodes()

    def get_max_expansion(self):
        return self.max_expansion

    def get_vn_id(self):
        return self.vn_id

    def set_vn_id(self, vn_id):
        self.vn_id = vn_id

    def get_sn_id(self):
        return self.sn_id

    def set_sn_id(self, sn_id):
        self.sn_id = sn_id

    def get_chosen_ids(self):
        return self.chosen_ids

    def set_chosen_ids(self, chosen_ids):
        self.chosen_ids = chosen_ids

    def is_terminal(self):
        """check the state is a terminal state"""
        # 如果当前映射的虚拟节点为最后一个节点或者没有为当前虚拟节点找到可映射的底层节点，则到达终止状态
        if self.vn_id == self.vnr.number_of_nodes() - 1 or self.sn_id == -1:
            return True
        else:
            return False

    def compute_final_reward(self):
        """
        如果虚拟网络请求能够被成功映射，那么最终奖赏为收益减去成本；否则，最终奖赏为一个无穷小的值
        """

        node_map, link_map = {}, {}
        for i in range(self.vnr.number_of_nodes()):
            node_map.update({i: self.chosen_ids[i]})
        link_map = Network.find_path(self.sub, self.vnr, node_map)
        if len(link_map) == self.vnr.number_of_edges():
            requested = Evaluation.calculate_revenue(self.vnr)
            occupied = Evaluation.calculate_cost(self.vnr, link_map)
            return 1000 + requested - occupied
        else:
            return LIMIT

    def get_next_state_with_random_choice(self):
        """针对下一个虚拟节点，随机选择一个可映射的底层节点"""

        actions = []
        for i in range(self.sub.number_of_nodes()):
            if i not in self.chosen_ids and \
                    self.sub.nodes[i]['weight'] >= self.vnr.nodes[self.vn_id + 1]['weight']:
                actions.append(i)
        self.max_expansion = len(actions)
        if self.max_expansion > 0:
            random_choice = random.choice([action for action in actions])
        else:
            random_choice = -1

        next_state = copy.deepcopy(self)
        next_state.set_vn_id(self.vn_id + 1)
        next_state.set_sn_id(random_choice)
        next_state.set_chosen_ids(self.chosen_ids + [random_choice])
        return next_state


class Node:
    """
    蒙特卡罗树搜索的树结构的Node，包含了如下信息：
    父节点和直接点等信息，
    用于计算UCB的遍历次数和quality值，
    选择这个Node的State
    """

    def __init__(self):
        self.parent = None
        self.children = []

        self.visit_times = 0
        self.value = 0.0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def value_add_n(self, n):
        self.value += n

    def is_all_expand(self):
        if len(self.children) == self.get_state().get_max_expansion():
            return True
        else:
            return False

    def add_child(self, child_node):
        child_node.set_parent(self)
        self.children.append(child_node)


class MCTS:
    def __init__(self, computation_budget, exploration_constant):
        self.computation_budget = computation_budget
        self.exploration_constant = exploration_constant

    def run(self, sub, vnr):
        node_map = {}
        current_node = Node()
        init_state = State(sub, vnr)
        current_node.set_state(init_state)

        for vn_id in range(vnr.number_of_nodes()):
            # print(current_node.get_state().get_max_expansion())
            current_node = self.search(current_node)
            if current_node is None:
                break
            sn_id = current_node.get_state().get_sn_id()
            if sn_id == -1:
                break
            node_map.update({vn_id: [sn_id,vnr.nodes[vn_id]["weight"]]})
        if len(node_map) != len(vnr):
            node_map = {}
        return node_map

    def search(self, node):
        """实现蒙特卡洛树搜索算法:
        传入一个根节点，在有限的时间内根据之前已经探索过的树结构expand新节点和更新数据，然后只要返回exploitation最高的子节点。
        蒙特卡洛树搜索包含四个步骤，Selection、Expansion、Simulation、Backpropagation。
        前两步使用tree policy找到值得探索的节点。
        第三步使用default policy也就是在选中的节点上随机算法选一个子节点并计算reward。
        最后一步使用backup也就是把reward更新到所有经过的选中节点上。
        进行预测时，只需要根据Q值选择exploitation最大的节点即可，找到下一个最优的节点。
        """

        # Run as much as possible under the computation budget
        for i in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(node)
            if expand_node is None:
                break

            # 2. Random run to add node and get reward
            reward = self.default_policy(expand_node)

            # 3. Update all passing nodes with reward
            self.backup(expand_node, reward)

        # N. Get the best next node
        best_next_node = self.best_child(node, False)

        return best_next_node

    def tree_policy(self, node):
        """
        蒙特卡罗树搜索的Selection和Expansion阶段，传入当前需要开始搜索的节点（例如根节点），
        根据exploration/exploitation算法返回最好的需要expand的节点，注意如果节点是叶子结点直接返回。
        基本策略是:
        (1)先找当前未选择过的子节点，如果有多个则随机选。
        (2)如果都选择过就找权衡过exploration/exploitation的UCB值最大的，如果UCB值相等则随机选。
        """

        while not node.get_state().is_terminal():

            if node.is_all_expand():
                node = self.best_child(node, True)
                if node is None:
                    break
            else:
                next_node = self.expand(node)
                return next_node

        # Return the leaf node
        return node

    def default_policy(self, node):
        """
        蒙特卡罗树搜索的Simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward。
        注意输入的节点应该不是子节点，而且是有未执行的Action可以expand的。基本策略是随机选择Action
        """

        current_state = node.get_state()

        while not current_state.is_terminal():
            # Pick one random action to play and get next state(对应论文中的第80行)
            current_state = current_state.get_next_state_with_random_choice()

        if current_state.get_sn_id() == -1:
            return LIMIT
        else:
            return current_state.compute_final_reward()

    def expand(self, node):
        """
        输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。
        注意，需要保证新增的节点与其他节点Action不同。
        """

        tried_actions = [child_node.get_state().get_sn_id() for child_node in node.get_children()]

        new_state = node.get_state().get_next_state_with_random_choice()

        # 首先它是一个可映射的底层节点
        if new_state.get_sn_id() != -1:
            # 其次，它要和其他以扩展节点的Action不同，这里就是底层节点
            while new_state.get_sn_id() in tried_actions:
                new_state = node.get_state().get_next_state_with_random_choice()

        next_node = Node()
        next_node.set_state(new_state)
        node.add_child(next_node)
        return next_node

    def best_child(self, node, is_exploration):
        """
        使用UCB算法，权衡exploration和exploitation后选择得分最高的子节点，
        注意如果是预测阶段直接选择当前Q值得分最高的。
        """

        best_score = LIMIT
        best_child_node = None

        # Travel all child nodes to find the best one
        for child_node in node.get_children():

            # Ignore exploration for inference
            if is_exploration:
                c = self.exploration_constant
            else:
                c = 0.0

            # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
            left = child_node.get_value() / child_node.get_visit_times()
            right = math.log(node.get_visit_times()) / child_node.get_visit_times()
            score = left + c * math.sqrt(right)

            if score > best_score:
                best_child_node = child_node
                best_score = score

        return best_child_node

    def backup(self, node, reward):
        """
        蒙特卡洛树搜索的Backpropagation阶段，输入前面获取需要expand的节点和新执行Action的reward，
        反馈给expand节点和上游所有节点并更新对应数据。
        """

        # Update util the root node
        while node is not None:
            # Update the visit times
            node.visit_times_add_one()

            # Update the quality value
            if reward != LIMIT:
                node.value_add_n(reward)

            # Change the node to the parent node
            node = node.parent
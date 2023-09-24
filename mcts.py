import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, rule_manager):
        node = self._root
        result = 0

        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            result = rule_manager.make_move(action)
            # state.do_move(action)

        action_probs, leaf_value = self._policy(rule_manager)
        # end, winner = state.game_end()
        if result == 0:
            node.expand(action_probs)
        else:
            if result == -2:
                leaf_value = -float(rule_manager.end_game()[0]) * rule_manager.turn
            else:
                leaf_value = -float(result)

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3, is_shown=False):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        if is_shown:
            return acts, act_probs, self._root._Q, visits
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-10)

    def get_action(self, rule_manager, temp=1e-3, return_prob=False, is_shown=False):
        sensible_moves = rule_manager.available_move()
        move_probs = np.zeros(rule_manager.boardSize * rule_manager.boardSize + 1)
        if len(sensible_moves) > 0:
            if not is_shown:
                acts, probs = self.mcts.get_move_probs(rule_manager, temp)
            else:
                acts, probs, value, visits = self.mcts.get_move_probs(rule_manager, temp, is_shown=True)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts, p=0.95*probs + 0.05*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-10)
                # location = rule_manager.convert(move, is_pair=False)
                # location = board.move_to_location(move)
                # print(location)
            if is_shown:
                return move, move_probs, -value.item(), visits
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("Warning no available moves")

    def __str__(self):
        return "MCTS {}".format(self.player)




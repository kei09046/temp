from __future__ import print_function
import random
from collections import deque
from gui import *
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time


class TrainPipeline:
    def __init__(self, init_model=False, test_model=False, gpu=False, cnt=0):
        self.boardSize = RuleManager.boardSize
        self.neutral = RuleManager.neutral

        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 800  # origin 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # origin 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 50  # origin 50
        self.game_batch_num = 1500
        self.cnt = cnt  # for save load
        self.episode_len = 0
        self.prev_policy = None
        if init_model:
            self.policy_value_net = PolicyValueNet(model_file=init_model, use_gpu=gpu)
        else:
            self.policy_value_net = PolicyValueNet(use_gpu=gpu)
        if test_model:
            self.prev_policy = PolicyValueNet(model_file=test_model, use_gpu=gpu)
        else:
            self.prev_policy = PolicyValueNet(use_gpu=gpu)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=True)

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_prob[1:].reshape(self.boardSize, self.boardSize)))
                extend_data.append((equi_state, np.append(mcts_prob[0], np.flipud(equi_mcts_prob).flatten()), winner))

            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, np.append(mcts_prob[0], np.flipud(equi_mcts_prob).flatten()), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = Game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            # print(play_data)
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        return

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        # print(winner_batch)
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        loss = -1
        entropy = -1
        new_v = None

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                             self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))

        return loss, entropy

    def policy_evaluate(self, n_games=30, process_type="single"):
        current_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        past_player = MCTSPlayer(self.prev_policy.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        win_cnt = 0.0

        if process_type == "multi":
            with Pool(processes=mp.cpu_count() - 1, maxtasksperchild=5) as p:
                for i in range(n_games):
                    win_cnt += p.apply_async(Game.start_play, (current_player, past_player),
                                             dict(start_player=i % 2, is_shown=True)).get()

        elif process_type == "single":
            for i in range(n_games):
                win_cnt += Game.start_play(current_player, past_player, start_player=i % 2, is_shown=True)

        win_ratio = 1.0 * win_cnt / n_games
        print(win_ratio)
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + self.cnt, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                if (i + self.cnt + 1) % self.check_freq == 0:
                    print("current self-play-batch: {}".format(i + self.cnt))
                    self.policy_value_net.save_model(f'./weights/model{i + self.cnt}.pt')
                    win_ratio = self.policy_evaluate()
                    if win_ratio > 0.55:
                        self.prev_policy = copy.deepcopy(self.policy_value_net)
                        print(f"new best model :{i + self.cnt}")
                    else:
                        self.policy_value_net = copy.deepcopy(self.prev_policy)
                    print("Saved")
        except KeyboardInterrupt:
            print('\n\rquit')


# RuleManager.boardSize = 3
# RuleManager.neutral = []
# RuleManager.penalty = 0

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model='./weights/model2599.pt',
                                      test_model='./weights/model2449.pt')
    # training_pipeline.run()
    start = time.time()
    training_pipeline.policy_evaluate(process_type="single")
    end = time.time()
    print(end - start)

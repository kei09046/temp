from tkinter import *
from policyvalue import *
from mcts import *
from PIL import Image, ImageTk

# 실제 게임이 실행되는 파일 GUI 담당이기도 함


class Game:
    def __init__(self, margin=50):
        self.rule = RuleManager()
        self.boardSize = RuleManager.boardSize
        self.neutral = RuleManager.neutral
        self.m = margin
        self.cord = [[(0, 0) for i in range(self.boardSize)] for j in range(self.boardSize)]
        self.delta = int((900 - self.m * 2) / (self.boardSize - 1))
        self.stone_size = int(self.delta * 0.8)
        self.cl = int(self.delta * 0.2)
        self.root = Tk()
        self.root.title("Great Kingdom")
        self.root.geometry("1200x1200")
        self.canvas = Canvas(self.root, width=900, height=900, bg="yellow")
        self.passButton = Button(self.root, text="pass")
        self.stones = []
        self.stones.append(
            ImageTk.PhotoImage(Image.open("images/b_stone.png").resize((self.stone_size, self.stone_size))))
        self.stones.append(
            ImageTk.PhotoImage(Image.open("images/w_stone.png").resize((self.stone_size, self.stone_size))))
        self.stones.append(
            ImageTk.PhotoImage(Image.open("images/neu_stone.png").resize((self.stone_size, self.stone_size))))
        self.turn = 0
        self.on_stone = [[False for i in range(self.boardSize)] for j in range(self.boardSize)]
        self.against_ai = False
        self.ai = None
        self.human_color = 0

    def create_board(self, ag_ai=False):
        self.canvas.pack()
        self.passButton.pack()
        self.passButton.configure(command=self.on_pass)

        for i in range(self.boardSize):
            self.canvas.create_line(i * self.delta + self.m, self.m, i * self.delta + self.m,
                               (self.boardSize - 1) * self.delta + self.m)
            self.canvas.create_line(self.m, i * self.delta + self.m, (self.boardSize - 1) * self.delta + self.m,
                               i * self.delta + self.m)

        for i in self.neutral:
            self.canvas.create_image(self.m + i[0] * self.delta - self.stone_size / 2,
                                     self.m + i[1] * self.delta - self.stone_size / 2,
                        anchor=NW, image=self.stones[2])
            self.on_stone[i[0]][i[1]] = True

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                self.cord[i][j] = (self.m + i * self.delta, self.m + j * self.delta)

        self.against_ai = ag_ai
        self.canvas.bind("<Button-1>", self.on_click)
        if self.against_ai:
            self.canvas.bind("<ButtonRelease-1>", self.ai_make_move)
        self.root.mainloop()

    def on_click(self, event):
        if not self.against_ai or (self.human_color == self.turn):
            x_cord = int((event.x - self.m) / self.delta + 0.5)
            y_cord = int((event.y - self.m) / self.delta + 0.5)
            x_loc = self.m + x_cord * self.delta
            y_loc = self.m + y_cord * self.delta

            if x_cord < 0 or x_cord >= self.boardSize or y_cord < 0 or y_cord >= self.boardSize:
                return

            if abs(event.x - x_loc) < self.cl and abs(event.y - y_loc) < self.cl and not self.on_stone[x_cord][y_cord]:
                self.canvas.create_image(x_loc - self.stone_size / 2, y_loc - self.stone_size / 2,
                                         anchor=NW, image=self.stones[self.turn])

                self.turn = 1 - self.turn
                self.on_stone[x_cord][y_cord] = True

                res = self.rule.make_move(x_cord, y_cord, train_ai=self.against_ai)
                if res == 1 or res == -1 or res == -2:
                    self.on_end(res * self.rule.turn)
        return

    def on_pass(self):
        if not self.against_ai or (self.human_color == self.turn):
            if self.rule.make_move(-1, 0) == -2:
                self.on_end(-2)
                return

            self.turn = 1 - self.turn
            return

    # 0이면 흑 1이면 백
    def play_ai(self, color=0, init_model='./weights/model249.pt', level=1000):
        self.ai = MCTSPlayer(PolicyValueNet(model_file=init_model, use_gpu=True).policy_value_fn, c_puct=5,
                             n_playout=level)
        self.human_color = color
        self.create_board(ag_ai=True)

    def ai_make_move(self, event):
        if self.against_ai and self.human_color != self.turn:
            move, prob, value, visits = self.ai.get_action(self.rule, return_prob=True, is_shown=True, temp=1e-3)
            print(value, visits)
            if move == -1:
                print("pass")

            move = RuleManager.convert(move, is_pair=False)

            x_loc = self.m + move[0] * self.delta
            y_loc = self.m + move[1] * self.delta
            self.canvas.create_image(x_loc - self.stone_size / 2, y_loc - self.stone_size / 2,
                                     anchor=NW, image=self.stones[self.turn])
            self.turn = 1 - self.turn
            self.on_stone[move[0]][move[1]] = True
            res = self.rule.make_move(move[0], move[1], train_ai=True)
            if res == 1 or res == -1 or res == -2:
                self.on_end(res * self.rule.turn)

    def on_end(self, winner):
        if winner == 1:
            print("winner is black")
            return
        if winner == -1:
            print("winner is white")
            return
        result = self.rule.end_game()
        if result[0] == 1:
            print("winner is black", result[1])
            return
        if result[0] == -1:
            print("winner is white", result[1])
            return
        print("draw")
        return

    @staticmethod
    def start_self_play(player, is_shown=False, temp=1e-1):
        rule_manager = RuleManager()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(rule_manager, temp=temp, return_prob=True)
            states.append(rule_manager.current_state())
            mcts_probs.append(move_probs)
            current_players.append(rule_manager.turn)
            result = rule_manager.make_move(move, train_ai=True)
            # if is_shown:
            #     continue
            if result != 0:
                winners_z = np.zeros(len(current_players), dtype=float)
                if result != -2:
                    result *= rule_manager.turn
                else:
                    result = rule_manager.end_game()[0]
                winners_z[np.array(current_players) == result] = 1
                winners_z[np.array(current_players) != result] = -1
                player.reset_player()
                if is_shown:
                    print("winner :", result)
                return result, zip(states, mcts_probs, winners_z)

    @staticmethod
    def start_play(player1, player2, start_player=0, is_shown=False, temp=1e-1):
        rule_manager = RuleManager()
        t = [-1, 0, 1]
        if start_player == 0:
            player_list = [player1, player2]
        else:
            player_list = [player2, player1]

        while True:
            current_player = player_list[t[rule_manager.turn]]
            move = current_player.get_action(rule_manager, temp=temp)
            res = rule_manager.make_move(move, train_ai=True)

            if res == 0:
                continue
            if is_shown:
                print(rule_manager.seq)
            if res == -2:
                res = rule_manager.end_game()
                if res == 0:
                    print("draw")
                    return 0.5
                else:
                    if res[0] == 1 and start_player == 0:
                        # print("winner is ", "current player", "win by point")
                        return 1
                    if res[0] == -1 and start_player == 1:
                        # print("winner is ", "current player", "win by point")
                        return 1
                    if res[0] == 0:
                        # print("result is draw")
                        return 0.5
                    # print("winner is ", "opponent ", "win by point")
                    return 0
            if res * rule_manager.turn == 1 and start_player == 0:
                # print("winner is ", "current player", "win by capture")
                return 1
            if res * rule_manager.turn == -1 and start_player == 1:
                # print("winner is ", "current player", "win by capture")
                return 1

            # print("winner is ", "opponent ", "win by capture")
            return 0


if __name__ == '__main__':
    RuleManager.boardSize = 5
    RuleManager.neutral = [(2, 2)]
    # RuleManager.penalty = 0
    g = Game()
    # g.play_ai(color=0, init_model='./weights/model2599.pt', level=2500)
    g.create_board()

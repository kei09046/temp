import numpy as np
import copy


class RuleManager:

    Max = 2 ** 16 - 1
    boardSize = 9
    neutral = [(4, 4)]
    penalty = 2.5
    x_adj = [-1, 0, 1, 0]
    y_adj = [0, 1, 0, -1]

    def __init__(self, board_size=None, neutral=None, penalty=None):
        self.turn = 1
        self.pp = False
        self.territory = [0, 0]
        if board_size is not None:
            RuleManager.boardSize = board_size
        if neutral is not None:
            RuleManager.neutral = neutral
        if penalty is not None:
            RuleManager.penalty = penalty

        # 플레이어가 보는 판
        self.board = np.pad(np.zeros((self.boardSize, self.boardSize), dtype=int),
                            ((1, 1), (1, 1)), 'constant', constant_values=(RuleManager.Max, RuleManager.Max))
        # 돌의 사활 결정
        self.st_board = np.pad(np.zeros((self.boardSize, self.boardSize), dtype=int), ((1, 1), (1, 1)),
                               'constant', constant_values=(RuleManager.Max, RuleManager.Max))
        self.liberty_list = np.zeros(self.boardSize * self.boardSize, dtype=int)
        self.st_cnt = 0
        # 집 결정
        self.bound_board = np.pad(np.zeros((self.boardSize, self.boardSize), dtype=int),
                                 ((1, 1), (1, 1)), 'constant', constant_values=(-1, -1))

        # 1이면 흑집 -1이면 백집
        # 다른 판과 크기 맞춰주기 위해 +2
        self.terr_board = np.zeros((self.boardSize + 2, self.boardSize + 2), dtype=int)
        self.area_cnt = 0
        # 영역 계산 알고리즘(calc)에 필요
        self.eq_list = np.empty(self.boardSize * self.boardSize + 2, dtype=int)
        # 0 : 흑 돌과 인접한 칸이면 true 아니면 false
        self.cl = np.zeros((2, self.boardSize + 2, self.boardSize + 2), dtype=bool)
        # 학습 시 컴퓨터가 착수할 수 있는 영역 -1:pass
        self.available = list(range(self.boardSize * self.boardSize))
        self.available.append(-1)

        for i in RuleManager.neutral:
            self.available.remove(RuleManager.convert(i, is_pair=True))
        self.seq = []

        for cord in RuleManager.neutral:
            self.board[cord[0] + 1][cord[1] + 1] = RuleManager.Max
            self.st_board[cord[0] + 1][cord[1] + 1] = RuleManager.Max
            self.bound_board[cord[0] + 1][cord[1] + 1] = -1

    def current_state(self):
        ret = np.zeros((4, self.boardSize, self.boardSize), dtype=float)
        s = self.board[1:-1, 1:-1].astype(float)
        ret[1][s * self.turn < 0] = 1.0
        ret[0][s * self.turn > 0] = 1.0
        for n in self.neutral:
            ret[1][n[0]][n[1]] = 1.0
            ret[0][n[0]][n[1]] = 1.0
        if self.seq:
            last_move = self.seq[-1]
            ret[2][last_move[0]][last_move[1]] = 1.0
        ret[3][:, :] = self.turn
        return ret

    def available_move(self):
        return self.available

    # 흑 승: 1반환, 무승부: 0반환, 백 승: -1반환
    def end_game(self):
        b_score, w_score = self.score()
        diff = b_score - w_score - self.penalty
        if diff > 0:
            return 1, diff
        if diff == 0:
            return 0, 0
        return -1, -diff

    # 착수 시 승리 : 1반환, 패배 : -1반환, 지속 : 0반환 계가 : -2반환
    def make_move(self, x, y=None, train_ai=True):
        if (x == -1 or y == -1) and self.pp:
            self.seq.append((x, y))
            return -2
        if x == -1 or y == -1:
            self.seq.append((x, y))
            self.pp = True
            self.turn *= -1
            return 0
        if y is None:
            pair = RuleManager.convert(x, is_pair=False)
            return self.make_move(pair[0], pair[1])
        self.seq.append((x, y))

        if train_ai:
            self.available.remove(RuleManager.convert((x, y), is_pair=True))

        # 패딩된 행렬 연산 위해서 +1
        x += 1
        y += 1

        # 돌이 있는 곳에 착수했는지 체크
        if not (self.st_board[x][y] == 0):
            return 0

        # 상대 집에 착수했는지 체크
        if self.terr_board[x][y] * self.turn == -1:
            return -1

        self.pp = False

        # 돌의 활로를 계산, 잡힌 돌이 있는지 판단
        self.board[x][y] = self.turn
        empty_space = 0
        same_adj = set()
        diff_adj = set()

        for i in range(4):
            u = self.board[x + self.x_adj[i]][y + self.y_adj[i]]
            if u == 0:
                empty_space += 1
            elif u == RuleManager.Max:
                pass
            elif u * self.turn > 0:
                same_adj.add(self.st_board[x + self.x_adj[i]][y + self.y_adj[i]])
            else:
                diff_adj.add(self.st_board[x + self.x_adj[i]][y + self.y_adj[i]])

        # print(empty_space)
        s = len(same_adj)
        d = len(diff_adj)

        # 인접한 상대 돌의 활로 감소
        for i in range(d):
            temp = diff_adj.pop()
            self.liberty_list[temp] -= 1
            if self.liberty_list[temp] == 0:
                return 1

        # 인접한 내 돌이 없는 경우
        if s == 0:
            self.st_cnt += 1
            self.st_board[x][y] = self.st_cnt
            self.liberty_list[self.st_cnt] = empty_space
            m = self.st_cnt

        # 인접한 내 돌이 있는 경우
        else:
            m = min(same_adj)
            same_adj.discard(m)
            self.st_board[x][y] = m

            for i in range(self.boardSize):
                for j in range(self.boardSize):
                    if self.st_board[i + 1][j + 1] in same_adj:
                        self.st_board[i + 1][j + 1] = m

            for i in range(s - 1):
                temp = same_adj.pop()
                self.liberty_list[m] += self.liberty_list[temp]
                self.liberty_list[temp] = 0

            self.liberty_list[m] += empty_space - s

        if self.liberty_list[m] == 0:
            return -1

        # print(self.liberty_list)

        # 영역 갱신
        self.bound_board[x][y] = -1
        for i in range(4):
            self.cl[int((1 - self.turn) / 2)][x + self.x_adj[i]][y + self.y_adj[i]] = True

        # 자기 집 안에 착수한 경우
        if self.terr_board[x][y] == self.turn:
            self.territory[int((-self.turn + 1)/2)] -= 1
            self.terr_board[x][y] = 0

        # 공배에 착수한 경우
        else:
            for i in range(4):
                nx = x + self.x_adj[i]
                ny = y + self.y_adj[i]
                if self.bound_board[nx][ny] != -1:
                    self.calc(self.bound_board[nx][ny], train_ai=train_ai)
                    break

        # print(self.territory)
        self.turn *= -1
        return 0

    def score(self):
        return self.territory[0], self.territory[1]

    def calc(self, where, train_ai=True):
        if where < 0:
            return
        if where > self.area_cnt:
            print("error")

        c = copy.deepcopy(self.bound_board)
        # print(where)
        cnt = 0
        adj = np.full((4, 6), False, dtype=bool)

        for i in range(len(self.eq_list)):
            self.eq_list[i] = i

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if c[i + 1][j + 1] == where:
                    if c[i][j + 1] == -1 and c[i + 1][j] == -1:
                        c[i + 1][j + 1] = cnt
                        cnt += 1

                    elif c[i][j + 1] == -1:
                        c[i + 1][j + 1] = c[i + 1][j]

                    elif c[i + 1][j] == -1:
                        c[i + 1][j + 1] = c[i][j + 1]

                    elif c[i][j + 1] == c[i + 1][j]:
                        c[i + 1][j + 1] = c[i][j + 1]

                    elif c[i][j + 1] < c[i + 1][j]:
                        c[i + 1][j + 1] = c[i][j + 1]
                        self.__modify(c[i + 1][j], c[i][j + 1])

                    else:
                        c[i + 1][j + 1] = c[i + 1][j]
                        self.__modify(c[i][j + 1], c[i + 1][j])

        dnt = 0
        for i in range(cnt):
            if i == self.eq_list[i]:
                self.eq_list[i] = dnt
                dnt += 1
            else:
                self.eq_list[i] = self.eq_list[self.eq_list[i]]

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.bound_board[i + 1][j + 1] > where:
                    self.bound_board[i + 1][j + 1] -= 1
                elif self.bound_board[i + 1][j + 1] == where:
                    self.bound_board[i + 1][j + 1] = self.area_cnt + self.eq_list[c[i + 1][j + 1]]

        if dnt > 4:
            print("dnt to big")
            print(self.board, "move sequence ", self.seq)

        # print(self.bound_board)

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                if self.bound_board[i + 1][j + 1] >= self.area_cnt:
                    d = self.bound_board[i + 1][j + 1] - self.area_cnt
                    # print(d)
                    if i == 0:
                        adj[d][0] = True
                    if j == 0:
                        adj[d][1] = True
                    if i == self.boardSize - 1:
                        adj[d][2] = True
                    if j == self.boardSize - 1:
                        adj[d][3] = True
                    if self.cl[0][i + 1][j + 1]:
                        adj[d][4] = True
                    if self.cl[1][i + 1][j + 1]:
                        adj[d][5] = True

        # print(cnt, dnt)
        t = [[False for i in range(2)] for j in range(4)]
        for i in range(dnt):
            t[i][0] = not (adj[i][0] and adj[i][1] and adj[i][2] and adj[i][3]) and not adj[i][5]
            t[i][1] = not (adj[i][0] and adj[i][1] and adj[i][2] and adj[i][3]) and not adj[i][4]

        for i in range(self.boardSize):
            for j in range(self.boardSize):
                u = self.bound_board[i + 1][j + 1] - self.area_cnt
                if u >= 0 and t[u][0]:
                    self.terr_board[i + 1][j + 1] = 1
                    self.territory[0] += 1
                    if train_ai:
                        self.available.remove(RuleManager.convert((i, j), is_pair=True))

                if u >= 0 and t[u][1]:
                    self.terr_board[i + 1][j + 1] = -1
                    self.territory[1] += 1
                    if train_ai:
                        self.available.remove(RuleManager.convert((i, j), is_pair=True))

        self.area_cnt += dnt - 1
        # print(self.bound_board)
        return

    def __modify(self, n, m):
        eqv = self.eq_list[n]
        if eqv == n:
            self.eq_list[n] = m
            return
        if eqv > n:
            self.eq_list[n] = m
            self.__modify(eqv, m)
            return
        self.eq_list[n] = min(eqv, m)
        self.__modify(max(eqv, m), min(eqv, m))
        return

    @staticmethod
    def convert(no, is_pair=False):
        if is_pair:
            return no[0] * RuleManager.boardSize + no[1]
        else:
            return no // RuleManager.boardSize, no % RuleManager.boardSize


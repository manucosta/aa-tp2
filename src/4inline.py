import random


class FourInLine:
    def __init__(self, playerR, playerY):
        self.board = [[' ',' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ',' '],[' ',' ',' ',' ',' ',' ']]
        self.last = [0] * 7
        self.playerR, self.playerY = playerR, playerY
        self.playerR_turn = random.choice([True, False])

    def play_game(self):
        self.playerR.start_game('R')
        self.playerY.start_game('Y')
        i = 0
        while True: #yolo

            if self.playerR_turn:
                player, char, other_player = self.playerR, 'R', self.playerY
            else:
                player, char, other_player = self.playerY, 'Y', self.playerR
            if player.breed == "human":
                self.display_board()
            column = player.move(self.last, self.board)-1
            if self.last[column] >= 6: # illegal move
                print "Perdi por cagon"
                player.reward(-99, self.board) # score of shame
                break
            self.board[column][self.last[column]] = char
            self.last[column] += 1
            if self.player_wins(char,column):
                player.reward(1, self.board)
                other_player.reward(-1, self.board)
                break
            if self.board_full(): # tie game
                print "ya PERDIMOS"
                player.reward(0.5, self.board)
                other_player.reward(0.5, self.board)
                break
            other_player.reward(0, self.board)
            self.playerR_turn = not self.playerR_turn

    def player_wins(self, char, column):
        row = self.last[column]-1
        # Chequeamos columna
        if row >= 3:
          if self.board[column][row] == self.board[column][row-1] == self.board[column][row-2] == self.board[column][row-3]:
            print "ya gane por columnas", char
            print "ultima: ", column
            return True
        # Chequeamos fila
        counter = 0
        for col in xrange(7):
            if self.board[col][row] == char:
              counter += 1
              if counter == 4:
                print "ya gane por filas", char
                print "ultima: ", column
                return True
            else:
              counter = 0
        # TODO: Chequeamos diagonales

        return False

    def board_full(self):
      for i in self.last:
        if i < 6:
          return False
      return True

    def display_board(self):
        board = [list(i) for i in zip(*self.board)]
        print ' ',
        for i in range(len(board[1])):  # Make it work with non square matrices.
              print i+1,
        print
        for i, element in enumerate(board):
              print i+1, ' '.join(element)


class Player(object):
    def __init__(self):
        self.breed = "human"

    def start_game(self, char):
        print "\nNew game!"

    def move(self, last, board):
        return int(raw_input("Your move? "))

    def reward(self, value, board):
        print "{} rewarded: {}".format(self.breed, value)

    def available_moves(self, last):
        res = []
        for i in xrange(7):
          if last[i] < 6:
            res.append(i+1)
        return res 


class RandomPlayer(Player):
    def __init__(self):
        self.breed = "random"


    def start_game(self, char):
        pass

    def move(self, last, board):
        return random.choice(self.available_moves(last))

class QLearningPlayer(Player):
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q = {} # (state, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards

    def start_game(self, char):
        aux = [' ']*6
        self.last_board = [aux] * 7
        self.last_move = None

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q.get((state, action)) is None:
            self.q[(state, action)] = 1.0
        return self.q.get((state, action))

    def move(self, last, board):
        aux = board
        for t in xrange(0,len(aux)):
          aux[t] = tuple(aux[t])
        self.last_board = tuple(aux)
        actions = self.available_moves(last)

        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(actions)
            return self.last_move

        qs = [self.getQ(self.last_board, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]

        return actions[i]

    def reward(self, value, board):
        if self.last_move:
            self.learn(self.last_board, self.last_move, value, tuple(board))

    def learn(self, state, action, reward, result_state):
        prev = self.getQ(state, action)
        maxqnew = max([self.getQ(result_state, a) for a in self.available_moves(state)])
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)

player1 = QLearningPlayer()
player2 = QLearningPlayer()

for i in xrange(0,1):
    juego = FourInLine(player1, player2)
    juego.play_game()

    juego.display_board()

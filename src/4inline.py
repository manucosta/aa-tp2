import random


contraDiagonalInicio = [(0,3),(0,4),(0,5),(1,5),(2,5),(3,5)]#desde donde inicia de abajito. column, row
contraDiagonalFin = [(3,0),(4,0),(5,0),(6,0),(6,1),(6,2)]
diagonalInicio = [(3,5),(4,5),(5,5),(6,5),(6,4),(6,3)]
diagonalFin = [(0,2),(0,1),(0,0),(1,0),(2,0),(3,0)]

class FourInLine:
    def __init__(self, playerR, playerY):
        self.board = [[' ',' ',' ',' ',' ',' '],
                      [' ',' ',' ',' ',' ',' '],
                      [' ',' ',' ',' ',' ',' '],
                      [' ',' ',' ',' ',' ',' '],
                      [' ',' ',' ',' ',' ',' '],
                      [' ',' ',' ',' ',' ',' '],
                      [' ',' ',' ',' ',' ',' ']]
        self.last = [0] * 7
        self.playerR, self.playerY = playerR, playerY
        self.playerR_turn = random.choice([True, False])
        self.winner = ' '

    def play_game(self):
        self.playerR.start_game('R')
        self.playerY.start_game('Y')
        iter = 0
        while True: #yolo
            # Turn selection
            if self.playerR_turn:
                player, char, other_player = self.playerR, 'R', self.playerY
            else:
                player, char, other_player = self.playerY, 'Y', self.playerR
            if player.breed == "human":
                self.display_board()
            # Based on the state, select an action
            sel_column = player.move(self.last, self.board)-1
            # Check if the selected action is ilegal
            if self.last[sel_column] >= 6:
                print "Accion ilegal!"
                player.reward(-99, self.board, self.last) # score of shame
                break
            # Is not ilegal, so lets update
            self.board[sel_column][self.last[sel_column]] = char
            self.last[sel_column] += 1
            # Check if the actual player wins
            if self.player_wins(char, sel_column):
                player.reward(1, self.board,self.last)
                other_player.reward(-1, self.board, self.last)
                self.winner = char
                break
            # Tie game?
            if self.board_full():
                print "Empate"
                player.reward(0.5, self.board, self.last)
                other_player.reward(0.5, self.board, self.last)
                self.winner = "Tie"
                break
            other_player.reward(0, self.board, self.last)
            # Swicht of turn
            self.playerR_turn = not self.playerR_turn

    def player_wins(self, char, column):
        row = self.last[column]-1
        # Check by columns
        if row >= 3:
          if self.board[column][row] == self.board[column][row-1] == self.board[column][row-2] == self.board[column][row-3]:
            print "Gano el jugador", char, "por columna", column+1
            return True
        # Check by rows
        counter = 0
        for col in xrange(7):
            if self.board[col][row] == char:
              counter += 1
              if counter == 4:
                  print "Gano el jugador", char, "por fila", row+1
                  return True
            else:
              counter = 0
        counter = 0
        if row + column > 3 and row + column < 8:
            c = contraDiagonalInicio[row + column - 3][0]
            r = contraDiagonalInicio[row + column - 3][1]
            ctope = contraDiagonalFin[row + column - 3][0]
            rtope = contraDiagonalFin[row + column - 3][1]
            while c <= ctope and r >= rtope: #estas dos deberian dejar de cumplirse al mismo tiempo
                if self.board[c][r] == char:
                  counter += 1
                  if counter == 4:
                      print "Gano el jugador", char, "por contra diagonal", column + 1
                      return True
                else:
                  counter = 0
                r -= 1
                c += 1
        ounter = 0
        if row - column > -3 and row - column < 4:
            c = diagonalInicio[row - column + 2][0]
            r = diagonalInicio[row - column + 2][1]
            ctope = diagonalFin[row - column + 2][0]
            rtope = diagonalFin[row - column + 2][1]
            while c >= ctope and r >= rtope: #estas dos deberian dejar de cumplirse al mismo tiempo
                if self.board[c][r] == char:
                  counter += 1
                  if counter == 4:
                      print "Gano el jugador", char, "por diagonal", column + 1
                      return True
                else:
                  counter = 0
                r -= 1
                c -= 1
        # TODO: Check by diagonals

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
        for i in xrange(len(board), 0, -1):
              print i, ' '.join(board[i-1])


class Player(object):
    def __init__(self):
        self.breed = "human"

    def start_game(self, char):
        print "\nNew game!"

    def move(self, last, board):
        return int(raw_input("Your move? "))

    def reward(self, value, board, lastDiscs):
        pass
        #print "{} rewarded: {}".format(self.breed, value)

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
        # self.q = {} # (state, action) keys: Q values
        self.q = [] # [((state, action), Q_values)]
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards

    def start_game(self, char):
        self.last_board = [[' ',' ',' ',' ',' ',' '],
                          [' ',' ',' ',' ',' ',' '],
                          [' ',' ',' ',' ',' ',' '],
                          [' ',' ',' ',' ',' ',' '],
                          [' ',' ',' ',' ',' ',' '],
                          [' ',' ',' ',' ',' ',' '],
                          [' ',' ',' ',' ',' ',' ']]
        self.last_move = None

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        # TODO: Probar "pesimista": 0.0 initial values.
        return_val = None
        
        q_aux = [(s, a) for ((s, a), _) in self.q]
        if not (state, action) in q_aux:
            self.q.append(((state, action), 1.0))
        
        for ((s, a), q_val) in self.q:
            if (s, a) == (state, action):
                return_val = q_val
                break

        return return_val
            
        #if self.q.get((state, action)) is None:
        #    self.q[(state, action)] = 1.0
        #return self.q.get((state, action))

    def move(self, last, board):
        # aux = board
        # for t in xrange(0,len(aux)):
        #  aux[t] = tuple(aux[t])
        self.last_board = board
        actions = self.available_moves(last)

        if random.random() < self.epsilon: # explore!
            self.last_move = random.choice(actions)
            return self.last_move
        
        #print 'last board', self.last_board
        #print 'actions', actions
        qs = [self.getQ(self.last_board, a) for a in actions]
        #print 'qs', qs
        #TODO: implementar softmax como alternativa a max
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        self.last_move = actions[i]

        return actions[i]

    def reward(self, value, board, lastDiscs):
        if self.last_move:
            self.learn(self.last_board, lastDiscs, self.last_move, value, board)

    def learn(self, state, lastDiscs, action, reward, result_state):
        prev = self.getQ(state, action)
        qs = [self.getQ(result_state, a) for a in self.available_moves(lastDiscs)]
        if len(qs) == 0:
            maxqnew = 0
        else: 
            maxqnew = max(qs)
        for i, ((s, a), _) in enumerate(self.q):
            if (s,a) == (state, action):
                q_new = prev + self.alpha * ((reward + self.gamma*maxqnew) - prev)
                self.q[i] = ((s,a),q_new) 

playerR = QLearningPlayer()
playerY = RandomPlayer()
rwins = 0.0
ywins = 0.0
ties = 0.0

for i in xrange(0,1000):
    print "Epoch: ", i
    juego = FourInLine(playerR, playerY)
    juego.play_game()

    if juego.winner == 'R':
        rwins += 1
    elif juego.winner == 'Y':
        ywins += 1
    else:
        ties += 1

    print rwins
    print ywins
    print ties

    r_rate = rwins / (rwins + ywins + ties)
    print "Red's rate of wins: ", r_rate
    #juego.display_board()

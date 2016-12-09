from __future__ import division

import random
import numpy as np
#from joblib import Parallel, delayed
import multiprocessing
import math
import matplotlib.pyplot as plt

import hashlib
import cPickle as pickle

contraDiagonalInicio = [(0,3),(0,4),(0,5),(1,5),(2,5),(3,5)]#desde donde inicia de abajito. column, row
contraDiagonalFin = [(3,0),(4,0),(5,0),(6,0),(6,1),(6,2)]
diagonalInicio = [(3,5),(4,5),(5,5),(6,5),(6,4),(6,3)]
diagonalFin = [(0,2),(0,1),(0,0),(1,0),(2,0),(3,0)]

class FourInLine:
    def __init__(self, playerR, playerY, rows, columns):
        self.board = [ [ ' ' for _ in range(rows)] for _ in range(columns)]
        self.reverse_board = [ [ ' ' for _ in range(rows)] for _ in range(columns)]
        self.rows = rows
        self.columns = columns
        self.last = [0] * columns
        self.playerR, self.playerY = playerR, playerY
        self.winner = ' '
        self.end = False

    def play_game(self):
        self.playerR.start_game('R')
        self.playerY.start_game('Y')
        self.playerR_turn = np.random.choice([True, False])
        while True: #yolo
            # Turn selection
            if self.playerR_turn:
                player, char, other_player = self.playerR, 'R', self.playerY
            else:
                player, char, other_player = self.playerY, 'Y', self.playerR
            if player.breed == "human":
                self.display_board()
            # Based on the state, select an action
            sel_column = player.move(self.last, self.board, self.available_moves())
            # Check if the selected action is ilegal
            if self.last[sel_column] >= self.rows:
                print "Accion ilegal!"
                player.reward(-99, self.board, self.last, self.available_moves()) # score of shame
                break
            # Is not ilegal, so lets update
            self.board[sel_column][self.last[sel_column]] = char
            self.last[sel_column] += 1
            # Check if the actual player wins
            if self.player_wins(char, sel_column):
                self.end = True
                player.reward(10.0, self.board,self.last, self.available_moves())
                other_player.reward(-5.0, self.board, self.last, self.available_moves())
                self.winner = char
                break
            # Tie game?
            if self.board_full():
                #print "Empate"
                self.end = True
                player.reward(0.5, self.board, self.last, self.available_moves())
                other_player.reward(0.5, self.board, self.last, self.available_moves())
                self.winner = "Tie"
                break
            other_player.reward(0.0, self.board, self.last, self.available_moves())
            # Swicht of turn
            self.playerR_turn = not self.playerR_turn

    def player_wins(self, char, column):
        row = self.last[column]-1
        # Check by columns
        if row >= 3:
          if self.board[column][row] == self.board[column][row-1] == self.board[column][row-2] == self.board[column][row-3]:
            #print "Gano el jugador", char, "por columna", column+1
            return True
        # Check by rows
        counter = 0
        for col in xrange(self.columns):
            if self.board[col][row] == char:
                counter += 1
                if counter == 4:
                    #print "Gano el jugador", char, "por fila", row+1
                    return True
            else:
                counter = 0
        # Check by diagonals
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
                      #print "Gano el jugador", char, "por contra diagonal", column + 1
                      return True
                else:
                  counter = 0
                r -= 1
                c += 1
        counter = 0
        if row - column > -3 and row - column < 4:
            c = diagonalInicio[row - column + 2][0]
            r = diagonalInicio[row - column + 2][1]
            ctope = diagonalFin[row - column + 2][0]
            rtope = diagonalFin[row - column + 2][1]
            while c >= ctope and r >= rtope: #estas dos deberian dejar de cumplirse al mismo tiempo
                if self.board[c][r] == char:
                  counter += 1
                  if counter == 4:
                      #print "Gano el jugador", char, "por diagonal", column + 1
                      return True
                else:
                  counter = 0
                r -= 1
                c -= 1
        return False

    def board_full(self):
        for i in self.last:
            if i < 6:
                return False
        return True

    def available_moves(self):
        res = []
        if self.end:
            return res
        for i in xrange(self.columns):
            if self.last[i] < self.rows:
                res.append(i)
        return res 

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

    def move(self, last, board, moves):
        input = int(raw_input("Please enter your move: ")) 
        return input-1

    def reward(self, value, board, lastDiscs, moves):
        pass
        #print "{} rewarded: {}".format(self.breed, value)



class RandomPlayer(Player):
    def __init__(self):
        self.breed = "random"

    def start_game(self, char):
        pass

    def move(self, last, board, moves):
        return np.random.choice(moves)


class QLearningPlayer(Player):
    def __init__(self, epsilon=0.1, alpha=0.3, gamma=0.9, tau = 0.25):
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q = {} # (state, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards
        self.tau = tau # temperature
        self.iter = 0 # amount of matches played

    def start_game(self, char):
        self.last_board = None #the last board is generated when the player moves
        self.last_move = None
        if self.iter != 0 and self.iter % 15000 == 0:
            self.tau /= 1.6
        if self.tau < 0.005:
            self.tau = 0.005
        self.iter += 1

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        # TODO: Probar "pesimista": 0.0 initial values.

        if not self.q.has_key((state, action)):
            self.q[(state, action)] = 0.0
        
        return self.q[(state, action)]
            
    def move(self, last, board, moves):
        self.last_board = mutable2inmutable(board)
        actions = moves

        ### Epsilon greedy
        # if random.random() < self.epsilon: # explore!
        #     self.last_move = np.random.choice(actions)
        #     return self.last_move
        
        qs = [self.getQ(self.last_board, a) for a in actions]
        ### Softmax
        softqs = softmax(qs, self.tau)
        action = np.random.choice(actions, p=softqs)
        
        # maxQ = max(qs)
        # if qs.count(maxQ) > 1:
        #     # more than 1 best option; choose among them randomly
        #     best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
        #     i = np.random.choice(best_options)
        # else:
        #     i = qs.index(maxQ)

        self.last_move = action

        return action

    def reward(self, value, board, lastDiscs, moves):
        # Cuando caigo aca porque me llamaron con reward 0 moves son movimientos disponibles para mi. Pero cuando caigo aca con reward distinto de 0 es porque el juego se termino de una forma u otra. En ese caso no tiene sentido tener movimientos disponibles.
        if self.last_move:
            self.learn(self.last_board, lastDiscs, self.last_move, value, board, moves)

    def learn(self, state, lastDiscs, action, reward, result_state, moves):
        # Estoy usando moves como si fueran movimientos disponibles para mi rival, pero por el comentario que puse en reward, de hecho son movimientos para mi!!! (salvo que hubiera llegado a un estado terminal, en el cual ninguno de los dos deberia tener movimientos).

        prev = self.getQ(state, action)
        result_state = mutable2inmutable(result_state)
        qs = [self.getQ(result_state, a) for a in moves]
        if len(qs) == 0:
            otherqnew = 0
        else:
            softqs = softmax(qs, self.tau)
            otherqnew = np.random.choice(qs, p=softqs)            
        # other_player_actions = [a for a in self.available_moves(lastDiscs)]
        # if len(other_player_actions) == 0:
        #     randqnew = 0.0
        # else:
        #     other_action = np.random.choice(other_player_actions)
        #     result_state[other_action-1][lastDiscs[other_action-1]] = 'Y'
        #     result_state = mutable2inmutable(result_state)
        #     randqnew = self.getQ(result_state, other_action)
        # self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*randqnew) - prev)
        self.q[(state, action)] = prev + self.alpha * ((reward + self.gamma*otherqnew) - prev) 

def softmax(qs, tau):
    distr = []
    a = max(qs)
    expos = [np.exp(q/tau - a/tau) for q in qs]
    suma = np.log(sum(expos)) + a/tau
    for q in qs:
        log_sm = q/tau - suma
        distr.append(np.exp(log_sm))
    return distr
        
def mutable2inmutable(board):
    aux = []
    for l in board:
        l_aux = tuple(l)
        aux.append(l_aux)
    return tuple(aux)

def inmutable2mutable(board):
    aux = []
    for l in board:
        l_aux = list(l)
        aux.append(l_aux)
    return aux


experimento = open('Experimentos', 'w')
experimento.close()
iteraciones = 250000

for a in [0.3]:
    for g in [0.9]:
        for t in [4.0]:
                print "Experimentando con " + "Alpha: " + str(a) + " Tau: " + str(t) + " Gamma: " + str(g)
                experimento = open('Experimentos', 'a')
                experimento.write("Alpha: " + str(a) + " Tau: " + str(t )+ " Gamma: " + str(g) + "\n")
                experimento.close()
                # Initialize
                rwins = 0.0
                ywins = 0.0
                ties  = 0.0
                
                r_rates = []
                y_rates = []
                
                playerR = QLearningPlayer(tau=t)
                playerY = RandomPlayer()

                # Lets play
                results = []
                for i in xrange(iteraciones):
                    juego = FourInLine(playerR, playerY, 6, 7)
                    juego.play_game()
                    if juego.winner == 'R':
                        rwins += 1
                        results.append('R')
                    elif juego.winner == 'Y':
                        ywins += 1
                        results.append('Y')
                    else:
                        ties += 1
                        results.append('T')
                    if i%10000 == 0: 
                        print i, "iteraciones, ", "tau: ", playerR.tau," Rate = ", rwins/(rwins + ywins + ties) 
       
                # Original perfomance measurement
                r_rate = rwins/(rwins + ywins + ties)
                experimento = open('Experimentos', 'a')
                experimento.write(" Red's rate of wins: " + str(r_rate) + "\n")
                
                # Taking into account only the last 10% matches
                lasts = [results[x] for x in range(int(iteraciones-iteraciones/10), iteraciones)]
                lasts_rwins = len([x for x in lasts if x == 'R'])
                lasts_r_rate = lasts_rwins/(iteraciones/10)
                experimento.write(" Red's rate of wins taking into account only the last 10% matches: " + str(lasts_r_rate) + "\n")
                
                # Weighted sum with linear (exponential) growth
                '''
                factor = 0.5
                sum_r_rate = 0
                for result_index in range(len(results)):
                    if results[result_index] == 'R':
                        sum_r_rate += result_index # Habria que ver alguna manera de normalizar, da numeros muy grandes y se pierde nocion.
                experimento.write(" Red's rate of wins with a weighted sum with UNDEFINED growth: " + str(sum_r_rate) + "\n")
                experimento.close()
                
                '''
                
                # Plot
                # x = np.arange(0, iteraciones, 1)
                # plt.plot(x, r_rates, color = 'r', label='Player R')
                # plt.plot(x, y_rates, color = 'b', label='Player Y')
                
                # axes = plt.gca()
                # axes.set_xlim([0, iteraciones])    # x-axis bounds
                # axes.set_ylim([0, 1])              # y-axis bounds
                    
                # plt.title('Rates')
                # plt.xlabel('Match number')
                # plt.ylabel('Rate of wins')
                # legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
                    
                # plt.grid()
                # plt.show()
                #plt.savefig('grafico.pdf')


# Con 100.000 iteraciones y escogiendo acciones aleatoriamente:

    # Random VS Random (Baseline)
        # Red's rate of wins: 0.49626
        # Red's rate of wins taking into account only the last 10% matches: 0.4976
        # Red's rate of wins with a weighted sum with linear growth: 2.483.985.245

    # Random VS QLearning (Alpha: 0.3 Tau: 0.5 Gamma: 0.9)
        # Red's rate of wins: 0.49928
        # Red's rate of wins taking into account only the last 10% matches: 0.4963
        # Red's rate of wins with a weighted sum with linear growth: 2.505.317.718

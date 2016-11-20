from __future__ import division

import random
import numpy as np
#from joblib import Parallel, delayed
import multiprocessing
import math
from math import floor
import matplotlib.pyplot as plt

import hashlib
import cPickle as pickle


class FourInLine:
    def __init__(self, playerR, playerY, rows, columns):
        self.board = [ [ ' ' for _ in range(rows)] for _ in range(columns)]
        self.reverse_board = [ [ ' ' for _ in range(rows)] for _ in range(columns)]
        self.rows = rows
        self.columns = columns
        self.last = [0] * columns
        self.playerR, self.playerY = playerR, playerY
        self.winner = ' '
        self.contraDiagonalInicio = []#desde donde inicia de abajito. column, row
        self.contraDiagonalFin = []
        self.diagonalInicio = []
        self.diagonalFin = []
        self.inicializarDiagonales(rows, columns)


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
                player.reward(-99, self.board, self.last, self.reverse_board, self.available_moves()) # score of shame
                break
            # Is not ilegal, so lets update
            self.board[sel_column][self.last[sel_column]] = char
            if char == 'R':
                self.reverse_board[sel_column][self.last[sel_column]] = 'Y'
            else:
                self.reverse_board[sel_column][self.last[sel_column]] = 'R'
            self.last[sel_column] += 1
            # Check if the actual player wins
            if self.player_wins(char, sel_column):
                player.reward(1, self.board,self.last, self.reverse_board, self.available_moves())
                other_player.reward(-1, self.board, self.last, self.reverse_board, self.available_moves())
                self.winner = char
                self.display_board()
                print sel_column+1
                break
            # Tie game?
            if self.board_full():
                #print "Empate"
                player.reward(0.5, self.board, self.last, self.reverse_board, self.available_moves())
                other_player.reward(0.5, self.board, self.last, self.reverse_board, self.available_moves())
                self.winner = "Tie"
                break
            other_player.reward(0, self.board, self.last, self.reverse_board, self.available_moves())
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
        if row + column > 2 and row + column < self.rows-1 + self.columns-1 -2:
            c = self.contraDiagonalInicio[row + column - 3][0]
            r = self.contraDiagonalInicio[row + column - 3][1]
            ctope = self.contraDiagonalFin[row + column - 3][0]
            rtope = self.contraDiagonalFin[row + column - 3][1]
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
        if row - column > self.rows - self.columns - 2 and row - column < self.rows - self.columns + 2:
            estabilizadorDeIndice = int(floor(self.rows/2)-1)
            c = self.diagonalInicio[row - column + estabilizadorDeIndice][0]
            r = self.diagonalInicio[row - column + estabilizadorDeIndice][1]
            ctope = self.diagonalFin[row - column + estabilizadorDeIndice][0]
            rtope = self.diagonalFin[row - column + estabilizadorDeIndice][1]
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

    def inicializarDiagonales(self, rows, columns):
        self.contraDiagonalInicio = []#desde donde inicia de abajito. column, row
        self.contraDiagonalFin = []
        self.diagonalInicio = []
        self.diagonalFin = []
        if columns > 3 and rows > 3:
            for r in xrange(3, rows):
                self.contraDiagonalInicio.append((0,r))
            for c in xrange(1, columns-3):
                self.contraDiagonalInicio.append((c,rows-1))

            for c in xrange(3, columns):
                self.contraDiagonalFin.append((c,0))
            for r in xrange(1, rows-3):
                self.contraDiagonalFin.append((columns-1,r))

            for c in xrange(3,columns):
                self.diagonalInicio.append((c,rows-1))
            for r in xrange(rows-2, 2,-1):
                self.diagonalInicio.append((columns-1,r))

            for r in xrange(rows-1-3, 0,-1):#indices, desgracia de todo programador...
                self.diagonalFin.append((0,r))
            for c in xrange(0, columns-3):
                self.diagonalFin.append((c,0))



    def board_full(self):
        for i in self.last:
            if i < self.rows:
                return False
        return True

    def available_moves(self):
        res = []
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

    def reward(self, value, board, lastDiscs, reverse_state, moves):
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
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9, tau = 0.25):
        self.breed = "Qlearner"
        self.harm_humans = False
        self.q = {} # (state, action) keys: Q values
        self.epsilon = epsilon # e-greedy chance of random exploration
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards
        self.tau = tau # temperature

    def start_game(self, char):
        self.last_board = None #the last board is generated when the player moves
        self.last_move = None

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        # TODO: Probar "pesimista": 0.0 initial values.

        if not self.q.has_key((state, action)):
            self.q[(state, action)] = 1.0
        
        return self.q[(state, action)]
            

    def move(self, last, board, moves):
        self.last_board = mutable2inmutable(board)
        actions = moves

        ### Epsilon greedy
        #if random.random() < self.epsilon: # explore!
        #    self.last_move = np.random.choice(actions)
        #    return self.last_move
        
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

    def reward(self, value, board, lastDiscs, reverse_state, moves):
        if self.last_move:
            self.learn(self.last_board, lastDiscs, self.last_move, value, board, reverse_state, moves)

    def learn(self, state, lastDiscs, action, reward, result_state, reverse_state, moves):
        prev = self.getQ(state, action)
        reverse_state = mutable2inmutable(reverse_state)
        qs = [self.getQ(reverse_state, a) for a in moves]
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
        self.q[(state, action)] = prev + self.alpha * ((reward - self.gamma*otherqnew) - prev) 

def softmax(qs, tau):
    distr = []
    expos = [np.exp(q/tau) for q in qs]
    suma = sum(expos)
    for e in expos:
        distr.append(e / suma)
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

for a in [None]:
    for g in [None]:
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            print "Experimentando con " + "Alpha: " + str(a) + " Tau: " + str(t) + " Gamma: " + str(g)
            experimento = open('Experimentos', 'a')
            experimento.write("Alpha: " + str(a) + " Tau: " + str(t )+ " Gamma: " + str(g) + "\n")
            experimento.close()
            # Initialize
            rwins = 0.0
            ywins = 0.0
            ties  = 0.0

            playerR = QLearningPlayer(tau=t)
            playerY = RandomPlayer()
            
            # Lets play
            results = []
            for i in xrange(2):
                juego = FourInLine(playerR, playerY, 8, 8)
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
                    print i, "iteraciones, Rate = ", rwins/(rwins + ywins + ties) 
            '''
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
            
            factor = 0.5
            sum_r_rate = 0
            for result_index in range(len(results)):
                if results[result_index] == 'R':
                    sum_r_rate += result_index # Habria que ver alguna manera de normalizar, da numeros muy grandes y se pierde nocion.
            experimento.write(" Red's rate of wins with a weighted sum with UNDEFINED growth: " + str(sum_r_rate) + "\n")
            experimento.close()
            
            
            # Plot
            x = np.arange(0, iteraciones, 1)
            plt.plot(x, f1, color = 'r', label='Player R')
            plt.plot(x, f2, color = 'b', label='Player Y')
                     
            axes = plt.gca()
            axes.set_xlim([0, iteraciones])    # x-axis bounds
            axes.set_ylim([0, 1])              # y-axis bounds
            
            plt.title('Rate of wins', fontdict=titlefont)
            plt.xlabel('Match number', fontdict=labelfont)
            plt.ylabel('Rate of wins', fontdict=labelfont)

            plt.show()
            '''


# Con 100.000 iteraciones y escogiendo acciones aleatoriamente:

    # Random VS Random (Baseline)
        # Red's rate of wins: 0.49626
        # Red's rate of wins taking into account only the last 10% matches: 0.4976
        # Red's rate of wins with a weighted sum with linear growth: 2.483.985.245

    # Random VS QLearning (Alpha: 0.3 Tau: 0.5 Gamma: 0.9)
        # Red's rate of wins: 0.49928
        # Red's rate of wins taking into account only the last 10% matches: 0.4963
        # Red's rate of wins with a weighted sum with linear growth: 2.505.317.718

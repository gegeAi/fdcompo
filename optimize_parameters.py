import numpy as np
import pickle
import glob
import random
import sortedcontainers as sc
from cloud_as_function import CloudAsFunction

import pickle
import glob
import random
import numpy as np
import sortedcontainers as sc
from cloud_as_function import CloudAsFunction
import sys
from threading import Thread
import threading

class AtomicCounter:

    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        with self._lock:
            self.value += num
            return self.value


if __name__ == '__main__':
    import doctest
    doctest.testmod()


def add_game(used_set, player, game):
    if used_set[player] is None:
        used_set[player] = []
    used_set[player].append(game)


games = {}

print("<general>: loading games")

for filename in glob.iglob('.\small functions bt\*.dat'):
    with open(filename, 'rb') as f:
        games[filename[18:-4]] = pickle.load(f)

def compare_to(valid_function, knn_set, k):
    sorted_list = sc.SortedList()
    for player, functions in knn_set.items():
        for unique_function in functions:
            try:
                score = valid_function.custom_distance_with(unique_function)
                if score < CloudAsFunction.theoric_max:
                    sorted_list.add((score, player))
            except Exception as e:
                raise e
                
    player_count = {}
    try:
        for (score, player) in sorted_list[:k]:
            if player not in player_count.keys():
                player_count[player] = [0, 77777]
            player_count[player][0] += 1
            player_count[player][1] = min(score, player_count[player][1])
    except:
        return "No", 99999
        
    max_score = 666666
    max_count = 0
    max_player = ""

    for player, element in player_count.items():
        count = element[0]
        score = element[1]
        if max_count < count:
            max_player = player
            max_count = count
            max_score = score
        elif max_count == count:
            if max_score > score:
                max_player = player
                max_score = score
    return max_player, max_score


def split(game_set, folds=10, weights=np.asarray([87, 0, 0, 0, 1000, 1000,  1])):
    i = 0
    for i in range(folds):
        functions_train_set = dict.fromkeys(games.keys())
        functions_validation_set = dict.fromkeys(games.keys())
        
        for player, games_of_him in game_set.items():
            games_length = len(games_of_him)
            for index, game in enumerate(games_of_him):
                if index >= int(games_length*i/folds) and index < int(games_length*(i+1)/folds):
                    add_game(functions_validation_set, player, CloudAsFunction(game, weights))
                else:
                    add_game(functions_train_set, player, CloudAsFunction(game, weights))
        
        for player, games_of_him in functions_train_set.items():
            try:
                functions_train_set[player] = [CloudAsFunction.aggregate(games_of_him)]
            except Exception:
                print('cannot aggregate player, doesn\'t do anything then, pass')
                raise
                
        yield functions_train_set, functions_validation_set
    return
        
def split_valid(game_set, folds=4):
    i = 0
    for i in range(folds):
        functions_validation_set = dict.fromkeys(games.keys())
        
        for player, games_of_him in game_set.items():
            if games_of_him is None:
                continue
            games_length = len(games_of_him)
            for index, game in enumerate(games_of_him):
                if index >= int(games_length*i/folds) and index < int(games_length*(i+1)/folds):
                    add_game(functions_validation_set, player, game)
                else:
                    continue
                
        yield functions_validation_set
    return
    
    
def simulate(games_dict, weights=np.asarray([87, 35, 43, 50, 38, 40,  1])):
    print('<simulate>: simulate with parameters {}'.format(weights))
    correct = AtomicCounter(0)
    incorrect = AtomicCounter(0)
   
    class Computer(Thread):

        def __init__(self, functions_train_set, functions_validation_set, correct, incorrect):
            Thread.__init__(self)
            self.functions_train_set = functions_train_set
            self.functions_validation_set = functions_validation_set
            self.correct = correct
            self.incorrect = incorrect

        def run(self):
            for key, value in self.functions_validation_set.items():
                if value is None:
                    continue
                for one_game in value:
                    if one_game is None:
                        print("nononono")
                    try:
                        prediction, score = compare_to(one_game, self.functions_train_set, 1)
                        if key == prediction:
                            self.correct.increment()
                        else:
                            self.incorrect.increment()
                    except IndexError:
                        continue
    
    print("<simulate>: begin splitting")
    sets = split(games_dict, 10, weights)
    
    print("<simulate>: begin testing")
    for i, (functions_train_set, functions_validation_set) in enumerate(sets):
        computing = []
        valid_sets = split_valid(functions_validation_set)
        
        for sub_valid_set in valid_sets:
            computing.append(Computer(functions_train_set, sub_valid_set, correct, incorrect))
            computing[-1].start()
        for t in computing:
            t.join()
        break
    return correct.value / (correct.value + incorrect.value)

def recuit(t_init=1e2, mult_factor=0.98, t_stop=0.05, s0=np.asarray([1, 100, 100, 100, 500, 500,  1])):

    energy = 33 # energy with default parameters
    next_energy = 33
    t = t_init
    s = s0
    s_next = s
    current_success = 0
    next_success = 0
    while t > t_stop:
        proba = random.random()
        print("delta energy {}\ttemperature {}\tproba {}".format(energy-next_energy, t, np.exp((energy-next_energy)/t)))
        if proba < np.exp((energy-next_energy)/t) or next_success > current_success:
            print("parameters are conserved")
            s = s_next.copy()
            current_success = next_success
            energy = next_energy
        for i, val in enumerate(s[:-1]):
            s_next[i] = max(val + random.randint(-100, 100), 1)
        
        next_success = simulate(games, s_next)
        next_energy = 100 - next_success*100
        print("with parameters {}, accuracy is {}".format(s_next, next_success))
        t *= mult_factor

    return s

    
recuit()


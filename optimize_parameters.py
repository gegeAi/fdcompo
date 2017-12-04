import numpy as np
import pickle
import glob
import random
import sortedcontainers as sc
from cloud_as_function import CloudAsFunction


def add_game(used_set, player, game):
    if used_set[player] is None:
        used_set[player] = []
    used_set[player].append(game)


games = {}

for filename in glob.iglob('./small functions/*.dat'):
    with open(filename, 'rb') as f:
        games[filename[18:-4]] = pickle.load(f)


functions_train_set = dict.fromkeys(games.keys())
functions_validation_set = dict.fromkeys(games.keys())


def recuit(t_init=1e2, mult_factor=0.98, t_stop=0.05, s0=np.asarray([96, 29, 44, 47, 33, 40,  1])):

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
            s_next[i] = max(val + random.randint(-5, 5), 1)

        next_success = simulate(games, s_next)
        next_energy = 100 - next_success*100
        print("with parameters {}, accuracy is {}".format(s_next, next_success))
        t *= mult_factor

    return s


def compare_to(valid_function, knn_set, k):
    sorted_list = sc.SortedList()
    for player, functions in knn_set.items():
        for unique_function in functions:
            try:
                score = valid_function.custom_distance_with(unique_function)
                sorted_list.add((score, player))
            except Exception as e:
                raise e

    player_count = {}
    for (score, player) in sorted_list[:k]:
        if player not in player_count.keys():
            player_count[player] = [0, 100000000]
        player_count[player][0] += 1
        player_count[player][1] = min(score, player_count[player][1])

    max_score = 10000000000
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
    return max_player


def simulate(games_dict, weights):
    print("<simulate>: create model")
    for player, games_of_him in games_dict.items():
        for game in games_of_him:
            rdint = random.randint(0, 9)
            if rdint >= 9:
                add_game(functions_validation_set, player, CloudAsFunction(game, weights))
            else:
                add_game(functions_train_set, player, CloudAsFunction(game, weights))

    print("<simulate>: aggregate model")
    for player, games_of_him in functions_train_set.items():
        try:
            functions_train_set[player] = [CloudAsFunction.aggregate(games_of_him)]
        except Exception:
            print('cannot aggregate player, doesn\'t do anything then, pass')
            raise

    correct = 0
    incorrect = 0

    print("<simulate>: compute score")
    for key, value in functions_validation_set.items():
        if value is None:
            continue
        for one_game in value:
            prediction = compare_to(one_game, functions_train_set, 1)
            if key == prediction:
                correct += 1
            else:
                incorrect += 1

    return correct / (correct + incorrect)


recuit()


import pickle
import glob
import random
import numpy as np
import sortedcontainers as sc
import matplotlib.pyplot as plt
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


def simulate(games_dict, weights=np.asarray([87, 35, 43, 50, 38, 40,  1])):
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
            try:
                prediction = compare_to(one_game, functions_train_set, 1)
                if key == prediction:
                    correct += 1
                else:
                    incorrect += 1
            except IndexError:
                continue

    return correct / (correct + incorrect)


x = []
y = []

CloudAsFunction.step = 5

for i in range(120, 450, 5):
    CloudAsFunction.maximum_useful = i
    value = simulate(games)
    y.append(value)
    x.append(i)
    print("for interval [0, {}], success is : {}".format(i, value))


plt.plot(x, y)
plt.show()


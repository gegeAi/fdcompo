import pickle
import glob
import numpy as np
import sortedcontainers as sc
from cloud_as_function import CloudAsFunction
from multiprocessing import Process, Queue


def add_game(used_set, player, game):
    if used_set[player] is None:
        used_set[player] = []
    used_set[player].append(game)


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
    for (score, player) in sorted_list[:k]:
        if player not in player_count.keys():
            player_count[player] = [0, 77777]
        player_count[player][0] += 1
        player_count[player][1] = min(score, player_count[player][1])
        
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


def run(functions_train_set, functions_validation_set, q):
    correct = 0
    incorrect = 0
    for key, value in functions_validation_set.items():
        if value is None:
            continue
        for one_game in value:
            try:
                prediction, score = compare_to(one_game, functions_train_set, 1)
                if key == prediction:
                    correct += 1
                else:
                    incorrect += 1
            except IndexError:
                continue
    q.put((correct, incorrect))


def simulate(games_dict, weights=np.asarray([87, 0, 0, 0, 1000, 1000,  1])):
    correct = 0
    incorrect = 0

    q_list = []

    print("<simulate>: begin splitting")
    sets = split(games_dict, 10, weights)
    
    print("<simulate>: begin testing")
    for i, (functions_train_set, functions_validation_set) in enumerate(sets):
        print("<simulate>: compute score fold {}".format(i))
        computing = []
        valid_sets = split_valid(functions_validation_set)
        
        for sub_valid_set in valid_sets:
            q_list.append(Queue())
            computing.append(Process(target=run, args=(functions_train_set, sub_valid_set, q_list[-1])))
            computing[-1].start()
        for t, q in zip(computing, q_list):
            c, inc = q.get()
            correct += c
            incorrect += inc
            t.join()
        break
    return correct / (correct + incorrect)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    games = {}

    print("<general>: loading games")

    for filename in glob.iglob('.\small functions bt\*.dat'):
        with open(filename, 'rb') as f:
            games[filename[18:-4]] = pickle.load(f)

    bef_value = -1
    conseq_desc = 0
    for i in range(5, 51, 5):
        CloudAsFunction.window = i
        value = simulate(games, weights=np.asarray([1, 121, 121, 121, 101, 101,  1]))
        if value < bef_value:
            conseq_desc += 1
        else:
            conseq_desc = 0
        if conseq_desc >= 5:
            break
        bef_value = value
        print("for window = {}, success is : {}".format(i, value))
    print("for step = {}, success is : {}".format(i, value))



import pickle
import glob
import numpy as np
import sortedcontainers as sc
import pandas as pd
from cloud_as_function import CloudAsFunction

WEIGHTS = np.asarray([87, 35, 43, 50, 38, 40, 1])
def add_game(used_set, player, game):
    if used_set[player] is None:
        used_set[player] = []
    used_set[player].append(game)


def create_model(games_dict, weights=WEIGHTS):
    model = dict.fromkeys(games_dict.keys())

    for player, games_of_him in games_dict.items():
        for game in games_of_him:
            add_game(model, player, CloudAsFunction(game, weights))

    for player, games_of_him in model.items():
        try:
            model[player] = [CloudAsFunction.aggregate(games_of_him)]
        except Exception:
            print('cannot aggregate player, doesn\'t do anything then, pass')
            raise

    return model


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


games = {}

for filename in glob.iglob('./small functions bt/*.dat'):
    with open(filename, 'rb') as f:
        games[filename[18:-4]] = pickle.load(f)

functions_model = create_model(games)

with open('model_set.1nn', 'wb+') as f:
    pickle.dump(functions_model, f)

with open('./test functions/entry.dat', 'rb') as f:
    test_set = pickle.load(f)

with open('nametags.dat', 'rb') as f:
    nametags = pickle.load(f)

predictions = []
for test_game in test_set:
    player = compare_to(CloudAsFunction(test_game, WEIGHTS), functions_model, 1)
    print('player found : {}'.format(player))
    found = False
    for tag in nametags:
        if "-".join([tag.rsplit('/')[-4], tag.rsplit('/')[-3], tag.rsplit('/')[-2]]) == player[3:]:
            print('\tbattletag found : {}'.format(tag))
            if not found:
                predictions.append(tag)
                found = True

print(len(list(range(1, 341))))
print(len(predictions))

with open('predictions2.dat', 'wb+') as f:
    pickle.dump(predictions, f)

pd.DataFrame(data={'RowId': list(range(1, 341)), 'prediction': predictions}).to_csv('predictions.csv')

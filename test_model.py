import numpy as np
import pickle
import glob
import random
import sortedcontainers as sc
from scipy.spatial.distance import cdist

games = {}

for filename in glob.iglob('./small functions/*.dat'):
    with open(filename, 'rb') as f:
        games[filename[18:-4]] = pickle.load(f)

class CloudAsFunction(object):
    
    step = 5
    minimum_useful = 0
    maximum_useful = 200
    default_add_if_no_info = 1e-1
    
    def __init__(self, cloud, scale=True):
        self._cloud = cloud.copy()
        # temporary
        if scale:
            self._cloud[:, -3] *= 40
            self._cloud[:, -2] *= 10
            self._cloud[:, 1] *= 10
            self._cloud[:, 2] *= 10
            self._cloud[:, 3] *= 10
         
    def get_value_at(self, index):
        """
        return value at index if index is multiple of 5
        """
        for value_a in self._cloud:
            if value_a[-1] == index:
                return index, value_a[:-1]
        return self.get_max(), self._cloud[-1][:-1]
    
    def get_max(self):
        return int(self._cloud[-1][-1])
    
    def custom_distance_with(self, f2):
        maximum_x = min(f2.get_max(), self.get_max(), self.maximum_useful)
        distance = 0
        for i in range(self.minimum_useful, maximum_x, self.step):
            to_add = cdist(np.expand_dims(np.asarray(self.get_value_at(i)[1]), axis=0),
                              np.expand_dims(np.asarray(f2.get_value_at(i)[1]), axis=0), metric='cosine')[0, 0]
            if np.isnan(to_add):
                to_add = self.default_add_if_no_info
            distance += to_add
        return distance*self.step / (maximum_x - self.minimum_useful) if maximum_x > self.minimum_useful else 1000000

    @classmethod
    def aggregate(cls, list_of_functions):
        array = []
        copy_of_functions = list_of_functions.copy()
        for func in copy_of_functions:
            if func.get_max() < cls.maximum_useful:
                list_of_functions.remove(func)

        for i in range(cls.minimum_useful, cls.maximum_useful, cls.step):
            vector = np.asarray(list_of_functions[0].get_value_at(i)[1]).astype('float64')
            for func in list_of_functions[1:]:
                vector += np.asarray(func.get_value_at(i)[1])
            vector = [float(a) / float(len(list_of_functions)) for a in vector]
            vector.append(i)
            array.append(vector)
        return CloudAsFunction(array, False)

def add_game(used_set, player, game):
    if used_set[player] is None:
        used_set[player] = []
    used_set[player].append(game)


functions_train_set = dict.fromkeys(games.keys())
functions_validation_set = dict.fromkeys(games.keys())

for player, games_of_him in games.items():
    for game in games_of_him:
        rdint = random.randint(0, 9)
        if rdint >= 9:
            add_game(functions_validation_set, player, CloudAsFunction(game))
        else:
            add_game(functions_train_set, player, CloudAsFunction(game))

for player, games_of_him in functions_train_set.items():
    print(player)
    try:
        functions_train_set[player] = [CloudAsFunction.aggregate(games_of_him)]
    except Exception as e:
        print('cannot aggregate player, doesn\'t do anything then, pass')
        raise 

for player, games_of_him in games.items():
    print(player)
    try:
        games[player] = [CloudAsFunction.aggregate([CloudAsFunction(g) for g in games_of_him])]
    except Exception as e:
        print('cannot aggregate player, doesn\'t do anything then, pass')
        raise 

with open("game_model.knn", "wb+") as f:
    pickle.dump(games, f)

print(len(functions_train_set['AcerBly']))
print(len(functions_validation_set['AcerBly']))
print(len(games['AcerBly']))

def compare_to(valid_function, knn_set, k):
    sorted_list = sc.SortedList()
    for player, functions in knn_set.items():
        for function in functions:
            try:
                score = valid_function.custom_distance_with(function)
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

correct = 0
incorrect = 0

for key, value in functions_validation_set.items():
    print("Compare player {}".format(key))
    if value is None:
        continue
    for one_game in value:
        prediction = compare_to(one_game, functions_train_set, 1)
        print(prediction)
        if key == prediction:
            correct += 1
        else:
            incorrect += 1
        print(correct)
        print(incorrect)

print(correct)
print(incorrect)

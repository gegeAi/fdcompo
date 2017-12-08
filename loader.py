import pickle
import pandas as pd
import numpy as np

races = ['Protoss', 'Terran', 'Zerg']

csv_name = 'TRAIN.csv'
save_folder = 'small_functions_bt_elongated'


def vectorize_game(game):
    offset = 0
    if csv_name == 'TEST.csv':
        offset = -1
    output = np.zeros((1, 7))
    time = 0
    race = races.index(game[1 + offset]) * 100
    number_of_element_in_frame = 0
    buffer_output = np.zeros(output.shape)
    for val in game[2 + offset:]:

        # early stopping if game long enough
        if type(val) == float or time >= 500:
            return output

        sel = ((val == 's') * 2 - 1)
        base = ((val == 'Base') * 2 - 1)
        if 'hotkey' in val:
            hk10 = int(val[-2])
            hk01 = int(val[-1])
        else:
            hk01 = -1
            hk10 = -1

        mineral = ((val == 'SingleMineral') * 2 - 1)

        # if it is a timestamp, add the buffer as a new element
        if sel < 0 and base < 0 and hk10 < 0 and hk01 < 0:
            # reset counter
            number_of_element_in_frame = 0
            # add timestamp
            buffer_output[:, -1] = time
            # delete first line
            buffer_output = buffer_output[1:]
            buf_len = len(buffer_output)
            # elongate time
            for i, line in enumerate(buffer_output):
                line[-1] += 5*i/buf_len
            # append
            output = np.append(output, buffer_output, axis=0)
            # reset buffer
            buffer_output = np.zeros((1, 7))
            time += 5
        # if it is an action, add the vector to the buffer
        else:
            number_of_element_in_frame += 1
            buffer_output = np.append(buffer_output, np.asarray([[race, sel, base, mineral, hk10, hk01, 0]]), axis=0)

    return output


games = {}

for i, df in enumerate(pd.read_csv(csv_name, chunksize=1, header=None, engine='python', names=list(range(20000)))):
    print('Parsing line {}'.format(i))
    try:
        unique_game = []
        for value in dict(df).values():
            unique_game += list(value)
        if csv_name == 'TRAIN.csv':
            if unique_game[0] not in games.keys():
                games[unique_game[0]] = []
            games[unique_game[0]].append(vectorize_game(unique_game))
        else:
            if "entry" not in games.keys():
                games['entry'] = []
            games['entry'].append(vectorize_game(unique_game))
    except MemoryError:
        print('failed at game {}, break. Hope we can still save this'.format(i))
        raise

if csv_name == 'TRAIN.csv':
    for key, value in games.items():
        with open('{}\{}-{}-{}.dat'.format(save_folder, key.rsplit('/')[-4], key.rsplit('/')[-3], key.rsplit('/')[-2]), 'wb+') as f:
            pickle.dump(value, f)
else:
    for key, value in games.items():
        with open('{}\{}.dat'.format(save_folder, key), 'wb+') as f:
            pickle.dump(value, f)

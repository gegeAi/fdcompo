import numpy as np
from scipy.spatial.distance import cdist


class CloudAsFunction(object):
    step = 5
    window = 5
    theoric_max = 25000
    minimum_useful = 0
    maximum_useful = 250

    def __init__(self, cloud, weights=np.ones((7,))):
        self._cloud = cloud.copy()
        # temporary
        self._cloud *= weights

    def get_value_at(self, index):
        """
        return value at index if index is multiple of 5
        """
        count = 0
        vector = np.zeros((6,))
        for value_a in self._cloud:
            if abs(value_a[-1] - index) <= self.window/2:
                count += 1
                vector += value_a[:-1]
            if value_a[-1] - index > self.window/2:
                return index, vector / max(count, 1)
        return self.get_max(), self._cloud[-1][:-1]

    def get_max(self):
        return int(self._cloud[-1][-1])

    def custom_distance_with(self, f2):
        maximum_x = min(f2.get_max(), self.get_max(), self.maximum_useful)
        distance = 0
        if self._cloud[1][0] != f2._cloud[1][0]:
            return 5555555
        for i in range(self.minimum_useful, maximum_x, self.step):
            to_add = cdist(np.expand_dims(np.asarray(self.get_value_at(i)[1]), axis=0),
                           np.expand_dims(np.asarray(f2.get_value_at(i)[1]), axis=0), metric='euclidean')[0, 0]
            distance += to_add
            if distance / maximum_x > self.theoric_max:
                return distance / maximum_x
        return distance / maximum_x if maximum_x > self.minimum_useful else 4444444

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
        return CloudAsFunction(array)
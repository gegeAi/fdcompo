import numpy as np
from scipy.spatial.distance import cdist


class CloudAsFunction(object):
    step = 5
    minimum_useful = 0
    maximum_useful = 250
    default_add_if_no_info = 1e-1

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
            if abs(value_a[-1] - index) <= self.step/2:
                count += 1
                vector += value_a[:-1]
            if value_a[-1] - index > self.step/2:
                return index, vector / count
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
        return distance * self.step / (maximum_x - self.minimum_useful) if maximum_x > self.minimum_useful else 1000000

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
from itertools import product

import numpy as np


class hopfield:

    def __init__(self, w, x):
        self.w = w
        self.x = x
        self.dim = len(x)

    @staticmethod
    def activation_f(a):
        if a > 0:
            return 1
        else:
            return -1

    def update_async(self):
        for i in range(self.dim):
            self.x[i] = np.dot(self.w[i], self.x)
            self.x[i] = self.activation_f(self.x[i])


if __name__ == "__main__":
    w = np.array([[0, -2/3, 2/3], [-2/3, 0, -2/3], [2/3, -2/3, 0]])
    # print('wagi:\n', w, '\n')
    x = [1, 1, 1]
    h = hopfield(w, x)

    print(h.x)
    previous = []
    for i in range(3):
        h.update_async()
        print(h.x)
        if h.x == previous:
            break 
        previous = h.x.copy()

    print()
    for p in product(*[[-1, 1]]*3):
        h.x = list(p)
        h.update_async()
        same = '*' if list(p) == h.x else ''
        print(list(p), ' -> ', h.x, same)

from random import random

import numpy as np


class perceptron:
    
    def __init__(self, w=None, n=3):
        if not w is None:
            self.w = [np.array(w)]
        else:
            self.w=[[np.random.rand(n)]]

    def energia(self, x, d):
        y = self.policz_y(x)
        return (d - y)**2

    def energia_batch(self, l_x, l_d):
        e = 0
        for x, d in zip(l_x, l_d):
            e += self.energia(x, d)
        return e

    def policz_y(self, x):
        x = list(x) + [1]
        pre_y = np.sum(np.array(x) * self.w[-1])
        y = 1 if pre_y > 0 else 0
        return y

    def krok_uczenia(self, x, d, r=.1):
        y = self.policz_y(x)
        x = list(x) + [1]
        nowe_w = self.w[-1] + r * (d - y) * np.array(x)
        self.w.append(nowe_w)

    def ucz_batch_1(self, l_x, l_d, r=.1):
        for x, d in zip(l_x, l_d):
            self.krok_uczenia(x, d, r=r)

    def ucz_batch_2(self, l_x, l_d):
        dw = np.zeros(len(self.w[-1]))
        for x, d in zip(l_x, l_d):
            y = self.policz_y(x)
            if y > .5 and d == 0:
                dw -= list(x) + [1]
            if y <= .5 and d == 1:
                dw += list(x) + [1]
        self.w.append(self.w[-1] + dw)


    def trafnosc(self, l_x, l_d, verbose=False):
        wszystkie = len(l_x)
        prawidlowe = 0
        for x, d in zip(l_x, l_d):
            y = self.policz_y(x)
            ok = y > .5 and d == 1 or y <= .5 and d == 0
            if ok: prawidlowe += 1
        if verbose: 
            print('trafność: {}/{}'.format(prawidlowe, wszystkie))
        return prawidlowe
        

if __name__ == "__main__":
    
    training = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 1,
        (1, 1): 0,
    }
    l_x = list(training.keys())
    l_d = [training[x] for x in l_x]
    w = np.random.rand(3)
    r = .1
    # w = [-.12, .4, .65]
    # r = 1
    # w = [1, 1, 1]
    print('\nwagi początkowe:', w)

    print('\n  metoda 1')
    p = perceptron(w=w)

    p.trafnosc(l_x, l_d, verbose=True)
    for i in range(20):
        p.ucz_batch_1(l_x, l_d, r=r)
        # print(p.w[-1])
        if p.trafnosc(l_x, l_d, verbose=True) is len(l_x):
            break

    print('\n  metoda 2')
    p2 = perceptron(w=w)
    # print(p2.w, '\n')

    p2.trafnosc(l_x, l_d, verbose=True)
    for i in range(20):
        p2.ucz_batch_2(l_x, l_d)
        # print(p2.w[-1])
        if p2.trafnosc(l_x, l_d, verbose=True) is len(l_x):
            break

    print()
    
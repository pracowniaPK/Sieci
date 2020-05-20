from math import e

import numpy as np 


class Net:
    def __init__(self, w1, w2, fi, d_fi):
        self.w1 = np.array(w1)
        self.w2 = np.array(w2)
        self.fi = fi
        self.d_fi = d_fi

    def forward(self, x, verbose=False):
        self.x0 = np.ones(3)
        self.x0[1:3] = x

        self.z1 = np.dot(self.w1, self.x0)
        self.x1 = np.ones(3)
        self.x1[1:3] = self.fi(self.z1)

        self.z2 = np.dot(self.w2, self.x1)
        self.x2 = self.fi(self.z2)

        if verbose:
            print(x, '->', round(self.x2, 2))

        return self.x2

    def back(self, x, d, ro):
        # self.forward(x)

        d_x2 = (d - self.x2)  #  x2
        d_z2 = self.d_fi(self.z2) * d_x2
        d_w2 = ro * self.x1 * d_z2

        d_x1 = np.dot(self.w2[1:], d_z2)
        d_z1 = self.d_fi(self.z1) * d_x1
        d_w1 = ro * np.outer(d_z1, self.x0)

        return d_w1, d_w2
 
    def update_w(self, d_w1, d_w2):
        self.w2 += d_w2
        self.w1 += d_w1

if __name__ == "__main__":
    w1 = [
        [.86, -.16, .28],
        [.83, -.51, -.86]
    ]
    w2 = [.04, -.43, .48]

    def fi(t):
        return 1/(1+e**(-t))
    def d_fi(t):
        return fi(t) * (1 - fi(t))

    training_data = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]


    print('metoda energii cząstokwych\nprzed:')
    nn = Net(w1, w2, fi, d_fi)
    for t in training_data:
        nn.forward(t[:2], verbose=True)
    for _ in range(2000):
        for t in training_data:
            nn.forward(t[:2])
            d_w1, d_w2 =  nn.back(t[:2], t[2], 0.5)
            nn.update_w(d_w1, d_w2)
    print('\npo:')
    for t in training_data:
        nn.forward(t[:2], verbose=True)

    # print('w1', nn.w1)
    # print('w2', nn.w2)

    print('\nmetoda energii całkowitej\nprzed:')
    nn = Net(w1, w2, fi, d_fi)
    for t in training_data:
        nn.forward(t[:2], verbose=True)
    for _ in range(20000):
        d_w1 = np.zeros((2,3))
        d_w2 = np.zeros((3))
        for t in training_data:
            nn.forward(t[:2])
            d_w1_tmp, d_w2_tmp =  nn.back(t[:2], t[2], 0.5)
            d_w1 += d_w1_tmp
            d_w2 += d_w2_tmp
        nn.update_w(d_w1, d_w2)
    print('\npo:')
    for t in training_data:
        nn.forward(t[:2], verbose=True)


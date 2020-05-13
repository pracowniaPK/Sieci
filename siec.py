from math import e

import numpy as np 


class Net:
    def __init__(self, w1, w2, fi, d_fi):
        self.w1 = np.array(w1)
        self.w2 = np.array(w2)
        self.fi = fi
        self.d_fi = d_fi

    def forward(self, x):
        self.x0 = np.ones(3)
        self.x0[1:3] = x

        self.z1 = np.dot(self.w1, self.x0)
        self.x1 = np.ones(3)
        self.x1[1:3] = self.fi(self.z1)

        self.z2 = np.dot(self.w2, self.x1)
        self.x2 = self.fi(self.z2)

        return self.x2

    def back(self, x, d, ro):
        self.forward(x)

        d_x2 = (d - self.x2)  #  x2
        d_z2 = self.d_fi(self.z2) * d_x2
        d_w2 = ro * self.x1 * d_z2

        d_x1 = np.dot(self.w2[1:], d_z2)
        # print(self.z1)
        # print(self.d_fi(self.z1))
        # print(d_x1)
        d_z1 = self.d_fi(self.z1) * d_x1
        d_w1 = ro * np.outer(d_z1, self.x0)
        
        self.w2 += d_w2
        self.w1 += d_w1
        pass
        # print(d_w2)
        # print(d_w1)


if __name__ == "__main__":
    w1 = [
        [.86, -.16, .28],
        [.83, -.51, -.86]
    ]
    w2 = [.04, -.43, .48]
    # w1 = [
    #     [-6.06, -6.07, 2.45],
    #     [-4.89, -4.89, 7.29]
    # ]
    # w2 = [-9.8, 9.48, -4.47]
    def fi(t):
        return 1/(1+e**(-t))
    def d_fi(t):
        return e**(-t)/(1+e**(-t))**2

    training_data = [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]

    nn = Net(w1, w2, fi, d_fi)

    for t in training_data:
        print(nn.forward(t[:2]))
    for _ in range(5000):
        # print(nn.w1)
        # print(nn.w2)
        for t in training_data:
            # print(nn.forward(t[:2]))
            # print()
            nn.forward(t[:2])
            nn.back(t[:2], t[2], 0.5)
            # print(np.round(np.absolute(t[2]-nn.forward(t[:2])), decimals=3))
        # print()


    for t in training_data:
        print(nn.forward(t[:2]))



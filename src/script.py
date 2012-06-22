import sys
from pylab import *


def main():
    A = matrix(loadtxt('system_matrix.txt')[1:int(sys.argv[1]),
        1:int(sys.argv[1])])
    Ainv = linalg.inv(A)
    B = matrix(loadtxt('B.txt')[1:int(sys.argv[1]),
        :int(sys.argv[2])])

    j = matrix(zeros((int(sys.argv[2]),))).transpose()
    j[0] = 1.0
    j[3] = -1.0

    print B
    phi = Ainv * B * j
    savetxt('phi.txt', phi)


if __name__ == '__main__':
    main()

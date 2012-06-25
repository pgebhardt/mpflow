import sys
from pylab import *


def main():
    # load A
    A = matrix(loadtxt('system_matrix.txt')[:int(sys.argv[1]), :int(sys.argv[1])])

    # load B
    B = matrix(loadtxt('B.txt')[:int(sys.argv[1]), :int(sys.argv[2])])

    j = matrix(zeros((int(sys.argv[2]),))).transpose()
    j[4] = 1.0
    j[9] = -1.0

    phi = ones((int(sys.argv[1]), 1))
    Ainv = linalg.inv(A[1:, 1:])
    f = B[1:, :] * j
    phi[1:] = Ainv * (f - A[1:, 0])
    savetxt('phi.txt', phi)


if __name__ == '__main__':
    main()

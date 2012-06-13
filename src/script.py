import matplotlib.pyplot as plt
import numpy


def main():
    vertices = numpy.loadtxt('vertices.txt')
    plt.plot(vertices[:, 0], vertices[:, 1], 'd')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

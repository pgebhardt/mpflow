import matplotlib.pyplot as plt
import numpy


def main():
    vertices = numpy.loadtxt('vertices.txt')
    elements = numpy.loadtxt('elements.txt')

    for i in range(len(elements[:, 0])):
        plt.plot([vertices[elements[i, 0], 0], vertices[elements[i, 1], 0]],
            [vertices[elements[i, 0], 1], vertices[elements[i, 1], 1]], 'b')
        plt.plot([vertices[elements[i, 1], 0], vertices[elements[i, 2], 0]],
            [vertices[elements[i, 1], 1], vertices[elements[i, 2], 1]], 'b')
        plt.plot([vertices[elements[i, 2], 0], vertices[elements[i, 0], 0]],
            [vertices[elements[i, 2], 1], vertices[elements[i, 0], 1]], 'b')

    plt.plot(vertices[:, 0], vertices[:, 1], 'd')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

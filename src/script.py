import matplotlib.pyplot as plt
import numpy


def main():
    vertices = numpy.loadtxt('vertices.txt')[1:988, :2]
    phi = numpy.loadtxt('phi.txt')

    # create image
    image = numpy.zeros((201, 201))

    # fill image
    distance = 1.0 / 16.0
    dx = 2.0 / (201.0 - 1.0)
    dy = 2.0 / (201.0 - 1.0)

    for k in range(0, 987):
        iStart = (vertices[k, 0] - distance) * 100 + 100
        jStart = (vertices[k, 1] - distance) * 100 + 100
        iEnd = (vertices[k, 0] + distance) * 100 + 100
        jEnd = (vertices[k, 1] + distance) * 100 + 100

        for i in numpy.arange(iStart, iEnd):
            for j in numpy.arange(jStart, jEnd):
                x = i * dx - 1.0
                y = j * dy - 1.0

                if x ** 2 + y ** 2 <= 1.0 and \
                    (x - vertices[k, 0]) ** 2 + (y - vertices[k, 1]) ** 2 <= distance ** 2:
                    image[i, j] += phi[k] * (1.0 - \
                        numpy.sqrt((x - vertices[k, 0]) ** 2 + \
                            (y - vertices[k, 1]) ** 2) / distance)

    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()

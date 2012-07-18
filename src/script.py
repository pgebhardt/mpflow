from pylab import *
import sys


def main():
    image = loadtxt('output/image.txt')
    imshow(image)
    CS = contour(image, colors='k')
    clabel(CS, fontsize=9, inline=1)
    savefig('output/anregung-{}.png'.format(sys.argv[1]));


if __name__ == '__main__':
    main()

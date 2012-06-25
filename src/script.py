from pylab import *


def main():
    image = loadtxt('image.txt')
    imshow(image)
    CS = contour(image, colors='k')
    clabel(CS, fontsize=9, inline=1)
    show()


if __name__ == '__main__':
    main()

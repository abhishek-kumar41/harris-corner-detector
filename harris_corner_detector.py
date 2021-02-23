from pathlib import Path
import numpy
import skimage.io
import skimage.color
from matplotlib import pyplot
from scipy.signal import convolve2d
import cv2
from skimage.util import random_noise
from scipy import ndimage


def harris_corner():

    threshold = 1000
    degree_rotation = 0
    scaling_factor = 1
    salt_pepper_noise_amount = 0     # add 0.01

    # image_original = cv2.imread('image1.jpg')
    image_original = cv2.imread('image2.jpg')

    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    image = image_original

    image = cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)

    image = random_noise(image, mode='s&p', amount=salt_pepper_noise_amount)
    image = (image*255).astype(numpy.uint8)
    image = clip_image(image)

    image = ndimage.rotate(image, degree_rotation)

    height, width = image.shape

    gauss_kernel_size = 3
    sigma_value = 2
    gaussian_window = numpy.zeros((gauss_kernel_size, gauss_kernel_size), dtype=float)
    x = int(gauss_kernel_size / 2)
    y = int(gauss_kernel_size / 2)

    for m in range(-x, x + 1):
        for n in range(-y, y + 1):
            x1 = 2 * numpy.pi * (sigma_value ** 2)
            x2 = numpy.exp(-(m ** 2 + n ** 2) / (2 * sigma_value ** 2))
            gaussian_window[m + x, n + y] = x2 / x1

    image_smooth = convolve2d(image, gaussian_window, boundary='symm', mode='same')

    sobel_x = numpy.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

    sobel_y = numpy.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

    image_x = convolve2d(image_smooth, sobel_x)
    image_y = convolve2d(image_smooth, sobel_y)

    Ixx = numpy.square(image_x)
    Iyy = numpy.square(image_y)
    Ixy = numpy.multiply(image_x, image_y)
    Iyx = numpy.multiply(image_y, image_x)

    Ixx = convolve2d(Ixx, gaussian_window)
    Iyy = convolve2d(Iyy, gaussian_window)
    Ixy = convolve2d(Ixy, gaussian_window)
    Iyx = convolve2d(Iyx, gaussian_window)

    k = 0.04
    R = numpy.zeros(image.shape, dtype=float)

    for i in range(height):
        for j in range(width):
            M = numpy.array([[Ixx[i, j], Ixy[i, j]],
                             [Iyx[i, j], Iyy[i, j]]])
            R[i, j] = numpy.linalg.det(M) - k*numpy.square(numpy.trace(M))

    image = image.astype(numpy.uint8)

    corner_detected_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)

    for i in range(height):
        for j in range(width):
            if R[i, j] > threshold:
                corner_detected_image[i, j] = (255, 0, 0)

    pyplot.subplot(121)
    pyplot.imshow(image, cmap='gray')
    pyplot.title(f'Image with scaling = {scaling_factor}, degree = {degree_rotation}, '
                 f'salt pepper noise ={salt_pepper_noise_amount}')
    pyplot.subplot(122)
    pyplot.imshow(corner_detected_image, cmap='gray')
    pyplot.title(f'Harris Corner Detected Image, Threshold = {threshold}')
    pyplot.show()

    return


def clip_image(image):
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            if image[i, j] > 255:
                image[i, j] = 255
            if image[i, j] < 0:
                image[i, j] = 0
    return image


def main():

    harris_corner()

    return


if __name__ == '__main__':
    main()

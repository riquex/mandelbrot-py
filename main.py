#!/home/henrique/scripts/scripts_python/Tests/AreaRestrita/numpy/numpy/bin/python3
from numba import njit, prange, double, uint8, complex128, int32
from sys import exit
import cv2
import numpy as np

@njit(complex128(double, double, double, int32, int32, int32), fastmath=True)
def ojgrid(top, left, size, density, x, y):
    i = size/density
    r_x = left + i*x
    r_y = (top - i*y)*1j
    return r_x + r_y

@njit(uint8(complex128))
def mandelbrot(c: complex):
    real, imag, n, temp = 0., 0., 0, 0.
    for n in range(256):
        temp = real*real - imag*imag + c.real
        imag = 2 * imag * real + c.imag
        real = temp
        if real * real - imag * imag > 2: break
    return n

@njit(uint8[:,:](double, double, double, int32), fastmath=True, parallel=True)
def mSetImg(top, left, size, box)->np.ndarray:
    grid = np.empty((box, box), dtype=np.uint8)
    for y in prange(grid.shape[0]):
        for x in prange(grid.shape[1]):
            grid[y, x] = mandelbrot(ojgrid(top, left, size, box, x, y))
    return grid

@njit(double[:, :](int32, double, double, double, int32))
def expChunks(chunks, top, left, size, box):
    chunks = max(2, chunks)+1
    i = 0
    res = np.zeros((np.square(chunks-1), 4))
    tops = np.linspace(top, top-size, chunks)[:-1]
    lefts = np.linspace(left, left+size, chunks)[:-1]
    for y in range(tops.shape[0]):
        for x in range(lefts.shape[0]):
            res[i, 0] += tops[y]
            res[i, 1] += lefts[x]
            i += 1
    res[:, 2] += tops[0]-tops[1]
    res[:, 3] += box
    return res

def main(*args, **kwargs)->int:
    c, chunks, top, left, size, density = 0, 50, 1.5, -1.5, 3., 5000
    for t, l, s, d in expChunks(chunks, top, left, size, density):
        arr = mSetImg(t, l, s, int(d))
        cv2.imwrite(f'img/{c=}.png', arr); c += 1
        print(f'{t=}\t{l=}\t{c=}')
    return 0

if __name__ == '__main__':
    out = main()
    exit(out)

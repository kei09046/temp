import multiprocessing as mp
import time


def f(x):
    return x * x


if __name__ == '__main__':
    ret = 0
    t = time.time()
    with mp.Pool(processes=8) as p:
        for i in range(100000):
            ret += p.apply_async(f, (i,)).get()

    e = time.time()
    print(e - t, ret)
    ret = 0

    t = time.time()
    for i in range(100000):
        ret += f(i)

    e = time.time()
    print(e - t, ret)


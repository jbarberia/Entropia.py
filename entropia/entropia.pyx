#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport log, log2
from math import factorial

cdef double mean(list x):
    """Valor medio de una señal"""
    return sum(x) / len(x)


cdef double distance(list arr, double x):
    """Distancia media al cuadrado"""
    return mean([(y-x)**2 for y in arr])


def argsort(list arr):
    """Ordena una lista y devuelve sus indices"""
    to_sort = [(i, v) for i, v in enumerate(arr)]
    sort = sorted(to_sort, key=lambda x: x[1])
    return [i for i, _ in sort]


def ordinal_patterns(list x, int m=3, int t=1):
    """Devuelve un vector de probabilidades de cada patron posible"""
    partition = [x[i:i+1+(m-1)*t] for i in range(0, len(x)-(m-1)*t, t)]

    permutation = map(lambda x: tuple(argsort(x)), partition)

    counts = {}
    cdef double sum_w = 0
    for perm in permutation:
        counts[perm] = counts.get(perm, 0) + 1
        sum_w += 1

    p = [w / sum_w for w in counts.values()]
    return p


def weight_patterns(list x, int m=3, int t=1):
    """Devuelve un vector ponderado de probabilidades de cada patron posible"""
    partition = [x[i:i+1+(m-1)*t] for i in range(0, len(x)-(m-1)*t, t)]

    X = map(mean, partition)
    weight = map(distance, partition, X)

    permutation = map(lambda x: tuple(argsort(x)), partition)

    counts = {}
    cdef double sum_w = 0
    for perm, w in zip(permutation, weight):
        counts[perm] = counts.get(perm, 0) + w
        sum_w += w

    p = [w / sum_w for w in counts.values()]
    return p


def weight_entropy(list x, int m=3, int t=1):
    y = weight_patterns(x, m, t)
    return - sum(x * log(x) for x in y)


def bandt_and_pompe(list x, int m=3, int t=1):
    y = ordinal_patterns(x, m, t)
    return - sum(x * log2(x) for x in y)


def bandt_and_pompe_normal(list x, int m=3, int t=1):
    return bandt_and_pompe(x, m, t) / log2(factorial(m))


def renyi_entropy(list x, int q=1, int m=3, int t=1):
    if q == 1:
        return bandt_and_pompe(x, m, t)
    y = ordinal_patterns(x, m, t)
    return log2(sum(x**q for x in y)) / (1-q)
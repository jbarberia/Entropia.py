#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport log, log2
from math import factorial

cdef double mean(list x):
    """Valor medio de una señal"""
    return sum(x) / len(x)


cdef double distance(list arr, double x):
    """Desvio del arr con respecto al valor x (Distancia cuadratica media)"""
    return mean([(y-x)**2 for y in arr])


def argsort(list arr):
    """Devuelve los indices que resultan al ordenar arr"""
    to_sort = [(i, v) for i, v in enumerate(arr)]
    sort = sorted(to_sort, key=lambda x: x[1])
    return [i for i, _ in sort]


def ordinal_patterns(list x, int m=3, int t=1):
    """Devuelve un vector de probabilidades de cada patron posible (Omite patrones faltantes)"""
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
    """Entropia ponderada (No Normalizada)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    y = weight_patterns(x, m, t)
    return - sum(x * log(x) for x in y)


def bandt_and_pompe(list x, int m=3, int t=1):
    """Entropia de Bandt & Pompe (Shannon, Permutacion)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    y = ordinal_patterns(x, m, t)
    return - sum(x * log2(x) for x in y)


def bandt_and_pompe_normal(list x, int m=3, int t=1):
    """Entropia de Bandt & Pompe (Normalizada)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    return bandt_and_pompe(x, m, t) / log2(factorial(m))


def renyi_entropy(list x, int q=1, int m=3, int t=1):
    """Entropia de renyi
    
    arguments:
    x -- [list]: Senal a analizar
    q -- [int]: Exponente
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    if q == 1:
        return bandt_and_pompe(x, m, t)
    y = ordinal_patterns(x, m, t)
    return log2(sum(x**q for x in y)) / (1-q)


def complexity_entropy(list x, int m=3, int t=1):
    """ Devuelve una tupla (e, jsd, cjs) con los siguientes valores:
    e: Entropia de Bandt & Pompe Normalizada
    jsd: Divergencia de Jansen-Shannon
    cjs: Complejidad basada en la divergencia de Jansen-Shannon
    
    arguments:
    x -- [list]: Senal a analizar
    q -- [int]: Exponente
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    n = factorial(m)
    op = ordinal_patterns(x, m, t)
    h1 = -sum(y * log(y) for y in op)

    cdef double Q = - 1 / ((0.5+0.5/n)*log(0.5+(0.5/n)) + (0.5/n)*log(0.5/n)*(n-1) + 0.5*log(n))
    
    op2 = [(0.5*y + 0.5/n) for y in op] + ([0.5/n] * (n - len(op))) # Concatenacion de listas
    h2 = -sum(x * log(x) for x in op2)

    JSD = h2 - 0.5*h1 - 0.5*log(n)
    comp_JS = Q * JSD * h1/log(n)

    return (h1/log(n), JSD, comp_JS)


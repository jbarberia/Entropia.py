import numpy as np
from collections import Counter
from math import factorial


def embedding_vector(x, m, t):
    tmp = np.zeros((x.shape[0], m))
    for i in range(m):
        tmp[:, i] = np.roll(x, i*t)
    partition = tmp[(t*m-1):, :]
    return partition


def improve_ordinal_patterns(x, l=4, m=3, t=1):
    """Devuelve un vector de probabilidades para cada patron posible (Mapeo extendido), l es el nivel de discretizacion"""
    # Genero las particiones
    partition = embedding_vector(x, m, t)

    patterns = uq(x, partition, l)
    count = Counter(patterns)

    summa = sum(count.values())
    return np.array([x/summa for x in count.values()])


def uq(x, partition, l):
    x_min = x.min()
    x_max = x.max()

    delta = (x_max + x_min) / l

    patterns = []
    for part in partition:
        pattern = []
        for element in part:
            for i in range(0, l):
                lb = x_min if (i == 0) else delta * i
                ub = x_max + delta if (i+1 == l) else (delta * (i + 1) if (lb < delta * (i+1)) else lb + delta) 

                if lb <= element < ub:
                    pattern.append(i)
                    break
        patterns.append(tuple(pattern))
    return patterns


def ordinal_patterns(x, m=3, t=1):
    """Devuelve un vector de probabilidades de cada patron posible"""
    # Genero las particiones 
    partition = embedding_vector(x, m, t)

    # Obtengo las permutaciones
    permutation = np.argsort(partition)
    idx = _hash(permutation)

    # Suma de los pesos por permutacion
    counts = np.zeros(factorial(m))
    for i in range(counts.shape[0]):
        counts[i] = (idx == i).sum()

    p = counts[counts != 0] / np.sum(counts) 
    return p


def weight_patterns(x, m=3, t=1):
    """Devuelve un vector ponderado de probabilidades de cada patron posible"""
    # Genero las particiones 
    partition = embedding_vector(x, m, t)

    # Calculo pesos
    xm = np.mean(partition, axis=1)
    weight = np.mean((partition - xm[:, None])**2, axis=1)

    # Obtengo las permutaciones
    permutation = np.argsort(partition)
    idx = _hash(permutation)

    # Suma de los pesos por permutacion
    counts = np.zeros(factorial(m))
    for i in range(counts.shape[0]):
        counts[i] = sum(weight[i == idx])

    p = counts[counts != 0] / np.sum(counts) 
    return p


def _hash(x):
    # Funcion auxiliar: Devuelve un vector de indices de acuerdo a la permutacion (colwise):
    # Ej:
    #   3   3   2   2   1   1
    #   2   1   3   1   3   2
    #   1   2   1   3   2   3
    #   ---------------------
    #   6   5   4   3   2   1
    m, n = x.shape
    if n == 1:
        return np.zeros(m)
    return np.sum(np.apply_along_axis(lambda y: y < x[:, 0], 0, x), axis=1) * factorial(n-1) + _hash(x[:, 1:]) 

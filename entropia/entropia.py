from math import factorial, log, log2
import numpy as np


def ordinal_patterns(x, m=3, t=1):
    """Devuelve un vector ponderado de probabilidades de cada patron posible"""

    # Genero las particiones
    tmp = np.zeros((x.shape[0], m))
    for i in range(m):
        tmp[:, i] = np.roll(x, i*t)
    partition = tmp[(t*m-1):, :] 

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
    tmp = np.zeros((x.shape[0], m))
    for i in range(m):
        tmp[:, i] = np.roll(x, i*t)
    partition = tmp[(t*m-1):, :] 

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


def weight_entropy(x, m=3, t=1):
    """Entropia ponderada (No Normalizada)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    y = weight_patterns(x, m, t)
    return - sum(x * log2(x) for x in y) / factorial(m)


def bandt_and_pompe(x, m=3, t=1):
    """Entropia de Bandt & Pompe (Shannon, Permutacion)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    y = ordinal_patterns(x, m, t)
    return - sum(x * log2(x) for x in y)


def bandt_and_pompe_normal(x, m=3, t=1):
    """Entropia de Bandt & Pompe (Normalizada)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    return bandt_and_pompe(x, m, t) / log2(factorial(m))


def min_entropy(x, m=3, t=1):
    """Min entropia
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    """
    y = ordinal_patterns(x, m, t).max()
    return - np.log(y)


def renyi_entropy(x, q=1, m=3, t=1):
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


def tsallis_entropy(x, q=1, m=3, t=1):
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
    return - (1 - sum(x**q))/(1-q)


def complexity_entropy(x, m=3, t=1):
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

    Q = - 1 / ((0.5+0.5/n)*log(0.5+(0.5/n)) + (0.5/n)*log(0.5/n)*(n-1) + 0.5*log(n))
    
    op2 = [(0.5*y + 0.5/n) for y in op] + ([0.5/n] * (n - len(op))) # Concatenacion de listas
    h2 = -sum(x * log(x) for x in op2)

    JSD = h2 - 0.5*h1 - 0.5*log(n)
    comp_JS = Q * JSD * h1/log(n)

    return (h1/log(n), JSD, comp_JS)


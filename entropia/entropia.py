from math import factorial, log, log2 
from .patterns import ordinal_patterns, improve_ordinal_patterns, weight_patterns
import numpy as np


def _calculate_patterns(x: np.ndarray, m: int, t: int, pattern: str, **kwargs):
    if pattern.lower() in ["normal", "n"]:
        return ordinal_patterns(x, m, t)
    elif pattern.lower() in ["weight", "w"]:
        return weight_patterns(x, m, t)
    elif pattern.lower() in ["improve", "i"]:
        if "l" in kwargs:
            l = kwargs["l"]
            return improve_ordinal_patterns(x, m, t, l)
        else:
            raise ValueError("Especifique el valor del valor de discretizacion: \'L\'")
    else:
        raise ValueError("Seleccione una opcion [normal - weight - improve]") 


def shannon(x, m=3, t=1, pattern="n"):
    """Entropia de Shannon
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    pattern -- [string]: tipo de patron
    """
    y = _calculate_patterns(x, m, t, pattern)
    return - sum(x * log2(x) for x in y)


def shannon_normal(x, m=3, t=1, pattern="n"):
    """Entropia de Bandt & Pompe (Normalizada)
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    pattern -- [string]: tipo de patron
    """
    y = _calculate_patterns(x, m, t, pattern)
    return shannon(x, m, t) / log2(factorial(m))


def min_entropy(x, m=3, t=1, pattern="n"):
    """Min entropia
    
    arguments:
    x -- [list]: Senal a analizar
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    pattern -- [string]: tipo de patron
    """
    y = _calculate_patterns(x, m, t, pattern).max()
    return - np.log(y)


def renyi_entropy(x, q=1, m=3, t=1, pattern="n"):
    """Entropia de renyi
    
    arguments:
    x -- [list]: Senal a analizar
    q -- [int]: Exponente
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    pattern -- [string]: tipo de patron
    """
    if q == 1:
        return bandt_and_pompe(x, m, t)
    y = _calculate_patterns(x, m, t, pattern)
    return log2(sum(x**q for x in y)) / (1-q)


def tsallis_entropy(x, q=1, m=3, t=1, pattern="n"):
    """Entropia de renyi
    
    arguments:
    x -- [list]: Senal a analizar
    q -- [int]: Exponente
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    pattern -- [string]: tipo de patron
    """
    if q == 1:
        return shannon(x, m, t)
    y = _calculate_patterns(x, m, t, pattern)
    return - (1 - sum(x**q))/(1-q)


def complexity_entropy(x, m=3, t=1, pattern="n"):
    """ Devuelve una tupla (e, jsd, cjs) con los siguientes valores:
    e: Entropia de Bandt & Pompe Normalizada
    jsd: Divergencia de Jansen-Shannon
    cjs: Complejidad basada en la divergencia de Jansen-Shannon
    
    arguments:
    x -- [list]: Senal a analizar
    q -- [int]: Exponente
    m -- [int]: Dimension de embedding
    t -- [int]: Delay de muestreo
    pattern -- [string]: tipo de patron
    """
    n = factorial(m)
    op = _calculate_patterns(x, m, t, pattern)
    h1 = -sum(y * log(y) for y in op)

    Q = - 1 / ((0.5+0.5/n)*log(0.5+(0.5/n)) + (0.5/n)*log(0.5/n)*(n-1) + 0.5*log(n))
    
    op2 = [(0.5*y + 0.5/n) for y in op] + ([0.5/n] * (n - len(op))) # Concatenacion de listas
    h2 = -sum(x * log(x) for x in op2)

    JSD = h2 - 0.5*h1 - 0.5*log(n)
    comp_JS = Q * JSD * h1/log(n)

    return (h1/log(n), JSD, comp_JS)


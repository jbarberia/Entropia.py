import pytest
import entropia
from math import sin, pi

tol = 1e-3

# Ejemplos de:
# Permutation Entropy: A Natural Complexity Measure for Time Series
# Christoph Bandt and Bernd Pompe

def test_bandt_00():
    x = [4, 7, 9, 10, 6, 11, 3]
    e = entropia.bandt_and_pompe(x, m=2)

    assert abs(e - 0.918) <= tol


def test_bandt_01():
    x = [4, 7, 9, 10, 6, 11, 3]
    e = entropia.bandt_and_pompe(x, m=3)

    assert abs(e - 1.522) <= tol


# Ejemplos de:
# The comparison of several kinds of Permutation Entropy 
# Zhang Hong, Wang Long, He Shasha 

def test_bandt_normal_00():
    n = 2048
    time = [i/n for i in range(n)]
    x = [sin(2*pi*500*t) for t in time]

    e = entropia.bandt_and_pompe_normal(x, m=6)
    assert abs(e - 0.3755) <= tol


def test_bandt_normal_01():
    n = 2048
    time = [i/n for i in range(n)]
    x = [sin(2*pi*15*t) for t in time]

    e = entropia.bandt_and_pompe_normal(x, m=6)
    assert abs(e - 0.1579) <= tol


def test_bandt_normal_02():
    n = 2048
    time = [i/n for i in range(n)]
    x = [(1+0.5*sin(2*pi*5*t))*(sin(2*pi*50*t**2+2*pi*10*t)) for t in time]

    e = entropia.bandt_and_pompe_normal(x, m=6)
    assert abs(e - 0.2625) <= tol


# Ejemplos de:
# Codigos realizados por el PID:
# Análisis de señales basado en herramientas entrópicas y optimización en los planos informacionales.
# UTN-FRBA

def test_weight_00():
    x = [4, 7, 9, 10, 6, 11, 3]
    e = entropia.weight_entropy(x)

    assert abs(e - 0.9801) <= tol
from sympy import *
from sympy.abc import h, k
import numpy as np
from sympy.utilities.iterables import variations


def ZIsing(L, k, h):
# Calcula la función de partición a temperatura inversa k y  campo magnético h
    Z = 0
    M = 0
    # Iteramos a través de todas las posibles configuraciones
    # En cada iteración sumamos al valor anterior de Z, M
    for s in variations([1, -1], L**2, True):
        # Recolocamos s como una matriz LxL
        s = np.reshape(s, (L, L))
        Z += peso(s, k, h)
        M += np.abs(np.sum(s))*peso(s, k, h)
    return Z, M/Z/L**2


def peso(s, k, h):
# Calcula el peso estadístico de la configuración s a temperatura inversa k y 
# campo magnético h
# s es una matriz LxL con entradas +/- 1 que representa la configuración

    # Con roll desplazamos todos los spines 1 posición a la derecha
    # (condiciones de contorno cíclicas, el último pasa ser el primero)
    si = np.roll(s, 1, axis=0)
    # Lo mismo en el otro eje
    sj = np.roll(s, 1, axis=1)
    # Con einsum (convenio de sumación de Einstein) sumamos sobre
    # los 2 ejes de la matriz de spines s, primero sobre filas
    # (índice i) y sobre columnas (índice j)
    V = k*(np.einsum('ij,ij->', s, si)+np.einsum('ij,ij->', s, sj))
    # Tenemos V = k sum_ij s[i][j]s[i+1][j] + s[i][j]s[i][j+1]
    return exp(h*np.sum(s) + V)

# Calculamos la función de partición y distintos observables en L=2
L = 2
Z, M = ZIsing(L, k, h)
# La energía es la d/dk log(Z) evaluada en h = 0
E = diff(log(Z), k).subs(h, 0)/L**2
M = M.subs(h, 0)


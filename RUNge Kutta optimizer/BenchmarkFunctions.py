import numpy as np

def BenchmarkFunctions(F):
    D = 30

    if F == 'F1':
        fobj = F1
        lb = -100
        ub = 100
    elif F == 'F2':
        fobj = F2
        lb = -100
        ub = 100
    elif F == 'F3':
        fobj = F3
        lb = -100
        ub = 100
    elif F == 'F4':
        fobj = F4
        lb = -100
        ub = 100
    elif F == 'F5':
        fobj = F5
        lb = -100
        ub = 100
    elif F == 'F6':
        fobj = F6
        lb = -100
        ub = 100
    elif F == 'F7':
        fobj = F7
        lb = -100
        ub = 100
    elif F == 'F8':
        fobj = F8
        lb = -100
        ub = 100
    elif F == 'F9':
        fobj = F9
        lb = -100
        ub = 100
    elif F == 'F10':
        fobj = F10
        lb = -32.768
        ub = 32.768
    elif F == 'F11':
        fobj = F11
        lb = -100
        ub = 100
    elif F == 'F12':
        fobj = F12
        lb = -100
        ub = 100
    elif F == 'F13':
        fobj = F13
        lb = -600
        ub = 600
    elif F == 'F14':
        fobj = F14
        lb = -50
        ub = 50

    return lb, ub, D, fobj

# 单峰值函数
# Bent Cigar
def F1(x):
  z = x[0] ** 2 + 1e6 * sum(x[1:] ** 2)
  return z

# Power
def F2(x):
  D = len(x)
  f = [abs(x[i]) ** (i + 1) for i in range(D)]
  z = sum(f)
  return z

# Zakharov
def F3(x):
  z = sum(x ** 2) + (sum(0.5 * x)) ** 2 + (sum(0.5 * x)) ** 4
  return z

# Rosenbrock
def F4(x):
  D = len(x)
  ff = [100 * (x[i] - x[i + 1]) ** 2 + (x[i] - 1) ** 2 for i in range(D - 1)]
  z = sum(ff)
  return z

# Discus
def F5(x):
  z = 1e6 * x[0] ** 2 + sum(x[1:] ** 2)
  return z

# High Conditioned Elliptic
def F6(x):
  D = len(x)
  f = [((10 ** 6) ** ((i - 1) / (D - 1))) * x[i] ** 2 for i in range(D)]
  z = sum(f)
  return z

# 多峰值函数
# np.expanded Schaffer’s F6
def F7(x):
  D = len(x)
  f = [0.5 + (np.sin(np.sqrt(x[i] ** 2 + x[i + 1] ** 2)) ** 2 - 0.5) / (1 + 1e-3 * (x[i] ** 2 + x[i + 1] ** 2)) for i in range(D - 2)]
  z = sum(f) + 0.5 + (np.sin(np.sqrt(x[D-1] ** 2 + x[0] ** 2) ** 2 - 0.5)) / (1 + 1e-3 * (x[D-1] ** 2 + x[0] ** 2))
  return z

# Levy函数
def F8(x):
  D = len(x)
  w = [1 + (x[i] - 1) / 4 for i in range(D)]
  f = [(w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2) for i in range(D - 2)]
  np.append(w[D-1], 1 + (x[D-1] - 1) / 4)
  z = np.sin(np.pi * w[0]) ** 2 + sum(f) + (w[D-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[D-1]) ** 2)
  return z

# np.modified Schwefel‘s
def F9(x):
  D = len(x)
  y = [x[i] + 4.209687462275036e+002 for i in range(D)]
  f1 = [y[i] * np.sin(abs(y[i]) ** 0.5) for i in range(D) if abs(y[i]) < 500]
  f2 = [(500 - np.mod(y[i],500)) * np.sin(np.sqrt(abs((500 - np.mod(y[i],500))))) - ((y[i]-500) ** 2)/(10000*D) for i in range(D) if y[i] > 500]
  f3 = [(np.mod(abs(y[i]),500) - 500) * np.sin(np.sqrt(abs(np.mod(abs(y[i]),500) - 500))) - ((y[i]+500) ** 2)/(10000*D) for i in range(D) if y[i] < -500]
  z = 418.9829 * D - sum(f1 + f2 + f3)
  return z

# Ackley
def F10(x):
  D = len(x)
  z = -20 * np.exp(-0.2 * ((1 / D) * sum(x ** 2)) ** 0.5) - np.exp(1 / D * sum(np.cos(2 * np.pi * x))) + 20 + np.exp(1)
  return z

# weierstrass
def F11(x):
  D = len(x)
  x = x + 0.5
  a = 0.5
  b = 3
  kmax = 20
  c1 = a ** (np.arange(kmax + 1))
  c2 = 2 * np.pi * b ** (np.arange(kmax + 1))
  f = 0
  c = -w(c1, c2, 0.5)
  for i in range(D):
      f += w(c1, c2, x[:][i])
  z = f + c * D
  return z

def w(c1, c2, x):
  x_arr = np.asarray(x)
  if (x_arr.ndim > 1):
    y = np.zeros((x_arr.ndim, 1))
    for k in range(x_arr.ndim):
      y[k] = sum(c1 * np.cos(c2 * x_arr[k]))
  else:
    y = sum(c1 * np.cos(c2 * x))
  return y

# HappyCat
def F12(x):
  D = len(x)
  z = (abs(sum(x ** 2) - D) ** (1 / 4)) + (0.5 * sum(x ** 2) + sum(x)) / D + 0.5
  return z

def F13(x):
  D = len(x)
  z = sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt([i+1 for i in range(D)]))) + 1
  return z

def F14(x):
  D = len(x)
  z = (np.pi / D) * (10 * np.sin(np.pi * ((x[0] + 1) / 4)) ** 2 + sum([((x[i] - 1) / 4) ** 2 * (1 + 10 * np.sin(np.pi * ((x[i + 1] + 1) / 4)) ** 2) for i in range(D - 1)]) + (x[-1] / 4) ** 2) + sum(Ufun(x, 10, 100, 4))
  return z

def Ufun(x, a, k, m):
  o = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
  return o

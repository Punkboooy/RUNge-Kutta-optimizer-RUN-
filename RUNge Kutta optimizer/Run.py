import numpy as np
from initialization import initialization
from RungeKutta import RungeKutta

def RUN(nP, MaxIt, lb, ub, dim, fobj):
  Cost = np.zeros((nP, 1))                   # Record the Fitness of all Solutions
  X = initialization(nP, dim, ub, lb)        # Initialize the set of random solutions
  Xnew2 = np.zeros(dim)
  Convergence_curve = np.zeros(MaxIt)

  for i in range(nP):
      Cost[i] = fobj(X[i])                   # Calculate the Value of Objective Function

  Best_Cost, Best_X = np.min(Cost), X[np.argmin(Cost)]   # Determine the Best Solution
  Convergence_curve[0] = Best_Cost

  # Main Loop of RUN
  it = 1
  while it < MaxIt:
      it += 1
      f = 20 * np.exp(-(12 * (it / MaxIt)))
      Xavg = np.mean(X)
      SF = 2 * (0.5 - np.random.rand(nP)) * f

      for i in range(nP):
          ind_l = np.argmin(Cost)
          lBest = X[ind_l]

          A, B, C = RndX(nP, i)
          ind1 = np.argmin(Cost[[A, B, C]])

          gama = np.random.rand() * (X[i] - np.random.rand(dim) * (ub - lb)) * np.exp(-4 * it / MaxIt)
          Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
          DelX = 2 * np.random.rand(dim) * np.abs(Stp)

          # Determine Xb and Xw for using in Runge Kutta method
          if Cost[i] < Cost[ind1]:
              Xb = X[i]
              Xw = X[ind1]
          else:
              Xb = X[ind1]
              Xw = X[i]

          # Search Mechanism (SM) of RUN based on Runge Kutta Method
          SM = RungeKutta(Xb, Xw, DelX)

          L = np.random.rand(dim) < 0.5
          Xc = L * X[i] + (1 - L) * X[A]
          Xm = L * Best_X + (1 - L) * lBest

          vec = np.array([1, -1])
          flag = np.floor(2 * np.random.rand(dim) + 1).astype(int)
          r = vec[flag - 1]

          g = 2 * np.random.rand()
          mu = 0.5 + 0.1 * np.random.randn(dim)

          # Determine New Solution Based on Runge Kutta Method
          if np.random.rand() < 0.5:
              Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * (SM) + mu * (Xm - Xc)
          else:
              Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * (SM) + mu * (X[A] - X[B])

          # Check if solutions go outside the search space and bring them back
          FU = Xnew > ub
          FL = Xnew < lb
          Xnew = (Xnew * (~(FU + FL))) + ub * FU + lb * FL
          CostNew = fobj(Xnew)

          if CostNew < Cost[i]:
              X[i] = Xnew
              Cost[i] = CostNew

      # Enhanced solution quality (ESQ)
      if np.random.rand() < 0.5:
          EXP = np.exp(-5 * np.random.rand() * it / MaxIt)
          r = np.floor(Unifrnd(-1, 2, 1, 1))

          u = 2 * np.random.rand(dim)
          w = Unifrnd(0, 2, 1, dim) * EXP

          A, B, C = RndX(nP, i)
          Xavg = (X[A] + X[B] + X[C]) / 3

          beta = np.random.rand(dim)
          Xnew1 = beta * Best_X + (1 - beta) * Xavg

          for j in range(dim):
              if w[0][j] < 1:
                  Xnew2[j] = Xnew1[j] + r * w[0][j] * np.abs((Xnew1[j] - Xavg[j]) + np.random.randn())
              else:
                  Xnew2[j] = (Xnew1[j] - Xavg[j]) + r * w[0][j] * np.abs((u[j] * Xnew1[j] - Xavg[j]) + np.random.randn())

          FU = Xnew2 > ub
          FL = Xnew2 < lb
          Xnew2 = (Xnew2 * (~(FU + FL))) + ub * FU + lb * FL
          CostNew = fobj(Xnew2)

          if CostNew < Cost[i]:
              X[i] = Xnew2
              Cost[i] = CostNew
          else:
              if np.random.rand() < w[0][np.random.randint(dim)]:
                  SM = RungeKutta(X[i], Xnew2, DelX)
                  Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (SM + (2 * np.random.rand(dim) * Best_X - Xnew2))

                  FU = Xnew > ub
                  FL = Xnew < lb
                  Xnew = (Xnew * (~(FU + FL))) + ub * FU + lb * FL
                  CostNew = fobj(Xnew)

                  if CostNew < Cost[i]:
                      X[i] = Xnew
                      Cost[i] = CostNew
      
      # Determine the Best Solution
      Best_Cost = min(Best_Cost, Cost[i])
      Best_X = X[np.argmin(Cost)]

      # Save Best Solution at each iteration  
      Convergence_curve[it - 1] = Best_Cost
      print('it : {}, Best Cost = {}'.format(it, Best_Cost))

  return Best_Cost, Best_X, Convergence_curve

# A funtion to determine a random number with uniform distribution
def Unifrnd(a, b, c, dim):
  # a2 = a / 2
  # b2 = b / 2
  # mu = a2 + b2
  # sig = b2 - a2
  # z = mu + sig * (2 * np.random.rand(c, dim) - 1)
  z = np.random.uniform(a, b, (c, dim))
  return z

# A function to determine thress random indices of solutions
def RndX(nP, i):
  Qi = np.random.permutation(nP)
  Qi = Qi[Qi != i]
  A, B, C = Qi[0], Qi[1], Qi[2]
  return A, B, C

import numpy as np

# This function initialize the first population of search agents
def initialization(nP, dim, ub, lb):
  ub_arr = np.asarray(ub)
  boundary_no = ub_arr.ndim if ub_arr.ndim > 1 else 1    # number of boundaries

  # If the boundaries of atl variables are equal and user enter a signle number for both ub and lb
  if boundary_no == 1:
      X = np.random.rand(nP, dim) * (ub - lb) + lb
  # If each variable has a different lb and ub
  elif boundary_no > 1:
      for i in range(dim):
          ub_i = ub[i]
          lb_i = lb[i]
          X[:, i] = np.random.rand(nP) * (ub_i - lb_i) + lb_i

  return X

import numpy as np
from BenchmarkFunctions import BenchmarkFunctions
from Run import RUN
import matplotlib.pyplot as plt

nP = 50             # Number of Population
Func_name = 'F5'    # Name of the test function, range from F1-F14
MaxIt = 500         # Maximum number of iterations

# Load details of the selected benchmark function
lb, ub, dim, fobj = BenchmarkFunctions(Func_name)

Best_fitness, BestPositions, Convergence_curve = RUN(nP, MaxIt, lb, ub, dim, fobj)

# Draw objective space
plt.figure()
plt.semilogy(Convergence_curve, color='r', linewidth=4)
plt.title('Convergence curve')
plt.xlabel('Iteration')
plt.ylabel('Best fitness obtained so far')
plt.axis('tight')
plt.grid(False)
plt.box(True)
plt.legend(['RUN'])
plt.show()

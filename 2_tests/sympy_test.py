from sympy import *
from sympy.printing.mathml import mathml
init_printing(use_unicode=True) # allow LaTeX printing
import numpy as np

# Define the symbols
t = symbols('t', real=True, positive=True)
x = Function('x', real=True)
y = Function('y', real=True)
z = Function('z', real=True)

om1, om2, om3 = symbols('om1 om2 om3', real=True)
k = symbols('k', real=True, positive=True)
#k = sqrt(om1**2 + om2**2 + om3**2)

# Define the system of differential equations
eq1 = Eq(x(t).diff(t), -om3 * y(t) + om2 * z(t))
eq2 = Eq(y(t).diff(t), om3 * x(t) - om1 * z(t))
eq3 = Eq(z(t).diff(t), -om2 * x(t) + om1 * y(t))

# Solve the system
x0, y0, z0 = symbols('x0 y0 z0', real=True)

x0_from_rel = sqrt(1-z0**2-y0**2)

ics = {x(0): 0, y(0): 0, z(0): 1}

solution = dsolve((eq1, eq2, eq3), ics=ics)  #
print("General Solution:")
for sol in solution:
    #sol = sol.subs((om1**2 + om2**2 + om3**2), k**2)
    print(sol.lhs,'=', latex(re(sol.rhs).simplify()))

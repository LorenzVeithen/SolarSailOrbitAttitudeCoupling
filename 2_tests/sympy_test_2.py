from sympy import symbols, Function, Eq, dsolve, solve

# Define the symbols
t = symbols('t')
x = Function('x')(t)
y = Function('y')(t)

# Define real constants for the solutions
C1, C2 = symbols('C1 C2', real=True)

# Define the system of differential equations
eq1 = Eq(x.diff(t), x + y)
eq2 = Eq(y.diff(t), -x + y)

# Solve the system
solution = dsolve((eq1, eq2))

# Print the general solution
print("General Solution:")
for sol in solution:
    print(sol)

# Define initial conditions symbolically
x0, y0 = symbols('x0 y0', real=True)

# Substitute initial conditions into the general solution
ics = {x.subs(t, 0): x0, y.subs(t, 0): y0}

# Extract the right-hand side of the solutions
rhs_solution = [sol.rhs for sol in solution]

# Solve for the constants C1 and C2 in terms of initial conditions
constants = solve([eq.subs(ics) for eq in rhs_solution], (C1, C2))

# Substitute the solved constants back into the general solution
particular_solution = [sol.subs(constants) for sol in solution]

# Print the particular solution
print("\nParticular Solution in terms of initial conditions:")
for sol in particular_solution:
    print(sol)

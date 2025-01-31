import sympy as sp

# Define the variables
x, y = sp.symbols('x y')


def critical_energy(lam):
    return 0.68 / (6 * lam ** 2)


def unbound_criterion(alpha):
    f = 1 / 2 * (x ** 2 + y ** 2) + alpha * (x ** 2 * y - 1 / 3 * y ** 3)

    # Compute the partial derivatives
    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)

    # Solve the equations df/dx = 0 and df/dy = 0
    critical_points = sp.solve([df_dx, df_dy], (x, y))

    # Display the critical points
    print("Critical Points:")
    print(critical_points)

    saddle_point = 0

    for point in critical_points:
        x_val, y_val = point
        f_val = f.subs({x: x_val, y: y_val})
        print(f"Point: {point}, f(x, y) = {f_val}")
        saddle_point = max(saddle_point, f_val)

    return saddle_point


if __name__ == "__main__":
    unbound_criterion(0.09)

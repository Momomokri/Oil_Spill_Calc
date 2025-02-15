import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


oil_types = {
    "light brown": 0.002,
    "brown": 0.005,
    "dark brown": 0.008,
    "black": 0.01
}


V = float(input("Enter initial volume of oil spill (m^3): "))
print("Available oil colors for thickness calculation: ", ", ".join(oil_types.keys()))
oil_color = input("Enter oil color from the options above: ").lower()
while oil_color not in oil_types:
    oil_color = input("Invalid choice. Enter a valid oil color: ").lower()
T0 = oil_types[oil_color]

wind_speed = float(input("Enter wind speed (m/s): "))
wind_direction = float(input("Enter wind direction (degrees, 0-360): "))



def T(r, wind_speed, T0, k=0.1):
    wind_factor = np.exp(-wind_speed * 0.05)
    return T0 * np.exp(-k * r) * wind_factor



def volume_integral(R, wind_speed, T0, k=0.1):
    def integrand(r, theta):
        return T(r, wind_speed, T0, k) * r

    vol, _ = spi.dblquad(integrand, 0, 2 * np.pi, lambda _: 0, lambda _: R)
    return vol



R_guess = np.sqrt(V / (np.pi * T(0, wind_speed, T0)))


R_solution = fsolve(lambda R: volume_integral(R, wind_speed, T0) - V, R_guess)[0]
A_solution = np.pi * R_solution ** 2

print(f"Estimated Spill Radius: {R_solution:.2f} meters")
print(f"Estimated Spill Area: {A_solution:.2f} square meters")
print(f"Wind Speed Considered: {wind_speed} m/s, Direction: {wind_direction} degrees")



def velocity_field(x, y, wind_speed, wind_direction, current_speed=2):
    angle_rad = np.radians(wind_direction)
    vx = wind_speed * np.cos(angle_rad) * np.exp(-y ** 2 / 100)  # Wind affecting X-direction
    vy = wind_speed * np.sin(angle_rad) * np.exp(-x ** 2 / 100) + current_speed * np.sin(
        x / 10)
    return vx, vy


def oil_spill_boundary(t, state, wind_speed, wind_direction, current_speed):
    x, y = state
    vx, vy = velocity_field(x, y, wind_speed, wind_direction, current_speed)
    return [vx, vy]


num_particles = 100
theta = np.linspace(0, 2 * np.pi, num_particles)
x_init = R_solution * np.cos(theta)
y_init = R_solution * np.sin(theta)

t_eval = np.linspace(0, 5, 100)
x_final, y_final = [], []

for i in range(num_particles):
    sol = solve_ivp(oil_spill_boundary, [0, 5], [x_init[i], y_init[i]], t_eval=t_eval,
                    args=(wind_speed, wind_direction, 2))
    x_final.append(sol.y[0, -1])
    y_final.append(sol.y[1, -1])


plt.figure(figsize=(7, 7))
plt.fill(x_final, y_final, color='black', alpha=0.5, label='Oil Spill')  # Always plot in black
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Simulated Oil Spill Shape with Fluid Dynamics")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

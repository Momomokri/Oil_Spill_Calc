import numpy as np #Arrays and functions
import scipy.integrate as spi #Solve Integrals
import matplotlib.pyplot as plt #Plot the shape
from scipy.integrate import solve_ivp #Used to solved differential equations
from scipy.optimize import fsolve #Used to solve functions


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


#T(r) = T0 * e^-kr * e^-.05w
def T(r, wind_speed, T0, k=0.1):
    """
    Determines the thickness at any radius r while taking into account wind speed.
    r: radius of which to find the thickness
    wind_speed: wind speed in m/s
    T0: initial thickness
    k: spread factor
    """
    wind_factor = np.exp(-wind_speed * 0.05)
    return T0 * np.exp(-k * r) * wind_factor


def integrand(r, theta):
    """
    Determines the value to be put into the inner integral in "volume_integral"
    :param r: radius of which to find the thickness
    :param theta: Theta value for dθ
    :return: The inner integral's calculated value without bounds
    """
    return T(r, wind_speed, T0, .01) * r

def volume_integral(R, wind_speed, T0, k=0.1):
    """
    Determines the volume of the oil spill at an infinite number of points between 0 and 2pi
    :param R: A variable that represents the total radius of the oil spill
    :param wind_speed: wind speed in m/s
    :param T0: initial thickness
    :param k: spread factor
    :return: The volume taking into account variable thickness and wind speed
    """
    vol, e = spi.dblquad(integrand, 0, 2 * np.pi, lambda _: 0, lambda _: R) #lambda is used in several places to convert a variable into an expression (i.e R into R(θ))
    return vol



R_guess = np.sqrt(V / (np.pi * T(0, wind_speed, T0))) #R_guess creates a rough estimate of what the total radius could be


R_solution = fsolve(lambda R: volume_integral(R, wind_speed, T0) - V, R_guess)[0] #Solves for the total radius by trying to make "volume_integral" match the originally input volume
A_solution = np.pi * R_solution ** 2 #Solves for the total area

print(f"Estimated Spill Radius: {R_solution:.2f} meters")
print(f"Estimated Spill Area: {A_solution:.2f} square meters")
print(f"Wind Speed Considered: {wind_speed} m/s, Direction: {wind_direction} degrees")



def velocity_field(x, y, wind_speed, wind_direction, current_speed=2):
    """
    Returns a velocity vector to show how the oil particles move over time.
    :param x: The x coordinate from which the velocity is calculated
    :param y: y coordinate from which the velocity is calculated
    :param wind_speed: wind speed in m/s
    :param wind_direction: wind direction in degrees
    :param current_speed: The speed at which oil spreads on water
    :return: The x velocity vector and y velocity vector as variables
    """
    angle_rad = np.radians(wind_direction) #Converts wind_direction into radians
    vx = wind_speed * np.cos(angle_rad) * np.exp(-y ** 2 / 100) #Wind affecting X-direction
    vy = wind_speed * np.sin(angle_rad) * np.exp(-x ** 2 / 100) + current_speed * np.sin(
        x / 10) #Wind affecting Y-direction
    return vx, vy


def oil_spill_boundary(t, state, wind_speed, wind_direction, current_speed):
    """
    Gets the velocity vectors of all areas within the oil spill.
    :param t: Time in seconds
    :param state: the current x and y position of a point on the oil spill boundary
    :param wind_speed: wind speed in m/s
    :param wind_direction: wind direction in degrees
    :param current_speed: The speed at which oil spreads on water
    :return: The x velocity vector and y velocity vector as a list
    """
    x, y = state
    vx, vy = velocity_field(x, y, wind_speed, wind_direction, current_speed)
    return [vx, vy]


num_particles = 100
theta = np.linspace(0, 2 * np.pi, num_particles) #Makes an angle that can be evenly distributed in a circle between the number of particles
x_init = R_solution * np.cos(theta) #Initial X-position to start calculations from
y_init = R_solution * np.sin(theta) #Initial Y-position to start calculations from

t_eval = np.linspace(0, 5, 100) #Creates 100 evenly spaced time points
x_final = [] #An empty list to store all the x values
y_final = [] #An empty list to store all the y values

for i in range(num_particles):
    sol = solve_ivp(oil_spill_boundary, [0, 5], [x_init[i], y_init[i]], t_eval=t_eval,
                    args=(wind_speed, wind_direction, 2)) #Creates a list full of x and y values at a specific particle over a time interval
    x_final.append(sol.y[0, -1]) #Appends the X list from sol
    y_final.append(sol.y[1, -1]) #appends the y list from sol

#Graphs all the data
plt.figure(figsize=(7, 7))
plt.fill(x_final, y_final, color='black', alpha=0.5, label='Oil Spill')  # Always plot in black
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Simulated Oil Spill Shape with Fluid Dynamics")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

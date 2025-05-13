import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 1000
num_graphs = 5  
deltaT = 0.01
sigma = 0.6
mu=1
S0 = 1  # Initial value of S

# Time steps
time_steps = np.linspace(0, num_points * deltaT, num_points)

def simulate_brownian_motion():
    """
    Simulates a single path of Brownian motion.
    """
    S_values = [S0]  # Start at initial value S0
    deltaZ = np.random.normal(0, np.sqrt(deltaT), num_points)  # Random increments

    # Loop through time steps
    for i in range(num_points):
        deltaS = mu*deltaT*S_values[-1] + sigma*deltaZ[i]*S_values[-1]  # Change in stock price (deltaS)
        S_next = S_values[-1] + deltaS  # Update stock price
        S_values.append(S_next)
        
    return S_values
  
# Generate and plot multiple Brownian motion paths
plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
for i in range(num_graphs):
    S_values = simulate_brownian_motion()  # Simulate a path
    plt.plot(time_steps, S_values[:-1], label=f"GBM {i+1}")

# Plot formatting
plt.xlabel("Time (seconds)")
plt.ylabel("S")
plt.title(f"Simulated Geometric Brownian Motion ({num_graphs} Paths)")
plt.grid(True)
plt.legend()
plt.show()


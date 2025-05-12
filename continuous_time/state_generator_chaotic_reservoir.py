import numpy as np
import random
import pickle
import time
from tqdm import tqdm
#@title Functions for generating inputs and states

# Defining different reservoirs to experiment with

# h       - reservoir state
# W and u - reservoir weights
# x_t     - input
# dt      - time step

def chaotic_reservoir(h, beta, theta, n, gamma, x_t, dt):
    """
    Mackey-Glass equation 
    (equation 2 from https://en.wikipedia.org/wiki/Mackey%E2%80%93Glass_equations)
    tau must be >= 17 for it to be chaotic
    """
    

# function which compiles the steps to generate inputs and reservoir states
# input: reservoir function (optional, will default to tanh)
def generate_states_lorenz(reservoir=chaotic_reservoir,
                           num_steps=100000, dt=0.0001,
                           initial_conditions = (0., 1., 1.05)):
  def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

  xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
  xyzs[0] = initial_conditions  # Set initial values

  # x, y, z = [0] * 10001, [0] * 10001, [0] * 10001
  # x[0], y[0], z[0] = xyzs[0]

  # Step through "time", calculating the partial derivatives at the current point
  # and using them to estimate the next point
  print("Generating Inputs")
  for i in tqdm(range(num_steps)):
      xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

  
  # Parameters
  beta = 0.2
  gamma = 0.1
  tau = 23.0
  n = 10.0
  t_span = int(num_steps*dt)  # Time steps
  # dt = 0.01  # Time step size
  t = np.arange(0, t_span, dt)

  # Initial condition (you'll need to define this)
  h0 = 1.0
  h = np.zeros_like(t)
  h[0] = h0


  x_list = xyzs.T[0]
  
  # Numerical solution using Euler method
  for i in tqdm(range(1, len(t))):
      h[i] = h[i-1] + dt * (beta * h[i-int(tau/dt)] / (1 + h[i-int(tau/dt)]**n) - gamma * h[i-1] + x_list[i-1])

  return xyzs, h

if __name__ == "__main__":

    start_time = time.time()
    xyzs, h_list = generate_states_lorenz(chaotic_reservoir,
                                            num_steps=10000000, dt=0.0001,
                                            initial_conditions = (3, 3, 3))
    x_list = xyzs.T[0][2000:]
    y_list = xyzs.T[1][2000:]
    z_list = xyzs.T[2][2000:]
    h_x = h_list[1999:]
    print("Finished generating all states")
    print(time.time() - start_time, "seconds to run")

    print("saving states")
    with open('x_list.pkl', 'wb') as file:
        pickle.dump(x_list, file)

    with open('h_x_chaotic.pkl', 'wb') as file:
        pickle.dump(h_x, file)

    # # saving hidden state vs reservoir state
    # ax = plt.figure().add_subplot(projection='3d')

    # ax.plot(y_list[:1000000:100], z_list[:1000000:100], h_x[1:1000000:100], lw=0.5)
    # ax.set_xlabel("Y Axis")
    # ax.set_ylabel("Z Axis")
    # ax.set_zlabel("Reservoir State X")

    # plt.savefig("chaotic_to_noisy_limit_cycle_attractor.png")
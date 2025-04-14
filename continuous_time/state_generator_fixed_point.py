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

def fixed_point_reservoir(h, k, D, x_t, dt):
    noise = D*np.random.normal(0, dt)
    return h + (-k*h + x_t + noise)*dt

# function which compiles the steps to generate inputs and reservoir states
# input: reservoir function (optional, will default to tanh)
def generate_states_lorenz(reservoir=fixed_point_reservoir,
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

  random.seed(10)
  k = 1
  D = .1
  h = np.array([0])

  h_list = np.empty((num_steps + 2, 1))
  h_list[0] = h
  x_list = xyzs.T[0]

  print("Generating Reservoir States")
  for i in tqdm(range(len(x_list))):
    # see "defining different reservoirs" cell for more details on the reservoir
    h = reservoir(h, k, D, x_list[i], dt)
    h_list[i+1] = h

  return xyzs, h_list

if __name__ == "__main__":

    start_time = time.time()
    xyzs, h_list = generate_states_lorenz(fixed_point_reservoir,
                                            num_steps=10000000, dt=0.0001,
                                            initial_conditions = (3, 3, 3))
    x_list = xyzs.T[0][2000:]
    y_list = xyzs.T[1][2000:]
    z_list = xyzs.T[2][2000:]
    h_x = h_list.T[0][2000:]
    print("Finished generating all states")
    print(time.time() - start_time, "seconds to run")

    print("saving states")
    with open('x_list.pkl', 'wb') as file:
        pickle.dump(x_list, file)

    with open('h_x_fixed_point.pkl', 'wb') as file:
        pickle.dump(h_x, file)

    # # saving hidden state vs reservoir state
    # ax = plt.figure().add_subplot(projection='3d')

    # ax.plot(y_list[:1000000:100], z_list[:1000000:100], h_x[1:1000000:100], lw=0.5)
    # ax.set_xlabel("Y Axis")
    # ax.set_ylabel("Z Axis")
    # ax.set_zlabel("Reservoir State X")

    # plt.savefig("chaotic_to_noisy_limit_cycle_attractor.png")
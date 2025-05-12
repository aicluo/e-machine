The files in this folder are for generating the MC/PC plots of the three reservoirs we have tried (fixed point, noisy limit cycle, chaotic).

Example usage:

If you would like to generate the MC/PC plot of the fixed point reservoir, for instance, you would run state_generator_fixed_point.py.
Parameters such as the weights of the reservoir, the number of steps, dt, and initial conditions can all be changed within the script.
Running the script will produce two output files: x_list.pkl, the inputs to the reservoir, and h_x_fixed_point.pkl, the reservoir states themselves.

Now, you can go to mc_pc_calculation.py, set the reservoir_type variable as "fixed_point", and then run the file. This will produce an image file of the MC/PC plot.

The notebooks in the notebooks folder include some preliminary parameter tuning on the chaotic reservoir which can be adapted for the other reservoirs,
as well as notebook-based implementations of the Python scripts

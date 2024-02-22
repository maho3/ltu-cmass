This package was originally developed for seeding large Nbody simulations. The base code is available here: https://www.h-its.org/2014/11/05/ngenic-code/#:~:text=NGenIC%20is%20an%20initial%20conditions,a%20homogeneously%20sampled%20periodic%20box.


Quijote uses it to generate initial conditions. The random seed used in the Quijote runs was equal to the latin hypercube ID. 

Calling the ngenic_white_noise executable can be like:

ngenic_white_noise num_mesh_1d num_part_1d random_seed white_noise_filename num_threads

Ludvig Doeser made adjustments to NGenIC to allow it to cutoff modes in the initial white noise field, to match Quijote's implementation of the latin hypercube HR (1024) set. In this usage, num_part_1d = num_mesh_1d//2 (512 and 1024 respectively).

However, in most instances and in the Quijote 512 LH, we want num_part_1d=num_mesh_1d.


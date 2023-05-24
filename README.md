# Jonah Kornberg 4/13/2021

Full Report:
https://drive.google.com/file/d/1xgbXWiKzZgHonwD_ZOCUkoltmTXoojJ9/view

Demo video:
https://www.youtube.com/watch?v=7iPG0C1Et6k


Instructions for running our project:
-Ensure Python3, Matplotlib, and Numpy are installed
To run with default settings:
python SeverSim.py, or python3 ServerSim.py

This will perform a single simulation using the FIFO cache replacement policy.
Below is a list of configuration options as well as their defaults. All unwritten arguments will run with default values

Example of running with arguments:
python ServerSim.py -u -s 5 -n 5000 -cs 200

-u
No arguments, this runs all cache replacement policies on the same set of requests. Ideal for comparing algorithms. Ignores the -t and -l flags as it uses fixed requests instead of time limit and all cache types. 

-s = 15 : integer
Access speed in mbps, describes the speed of accessing files from outside the institution

-i = 100 : integer
Institution speed in mbps, speed of accessing files within the network (from cache)

-n = 20000 : integer
Number of files in simulated server

-c = 1 : integer
Count of simulations, increase to regenerate files/probabilities and run multiple simulations

-l = 1000 : integer
Time limit in seconds, after limit no more requests are generated

-t = 0 : integer 0,1 or 2
Cache type. 0: FIFO, 1: Least Popular, 2: Best Fit

-r = 10000 : integer
Request count, ignored unless -u is set. Number of requests per simulation

-rt = .4 : float
Round trip time in seconds. 

-cs = 1000 : int
Cache size in mb

-rps = 10 : int
Requests per second


NOTE: below variables involve numpy pareto distribtion which has the PDF:
(var1 * (var2 ^ var1) ) / x ^ (var1 + 1)

Execution in code is calculated as: (np.random.pareto(var1) + 1) * var2

-sp1 = 2 : float
File size Pareto var1 in above equation

-sp2 = 1/2 : float
File size Pareto var2 in above equation

-pp1 = 2 : float
File probability Pareto var1 in above equation, but full results are normalized to sum to 1

-pp2 = 1/2 : float
File size Pareto var2 in above equation, but full results are normalized to sum to 1

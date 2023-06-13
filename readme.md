# PyMGA

A Python module conaining Modeling to Generate Alternatives and Modeling All Alternatives methods. 

## Instalation 

Clone the repository to the decired installation folder, e.g. 'src/PyMGA'  and install the package with one of the following approaches:

###### Install with pip

Execute the following command from the package folder, e.g. `src/PyMGA`

```
src/PyMGA> pip install PyMGA
Processing /PyMGA
Installing collected packages: PyMGA
Running setup.py install for PyMGA ... done
Successfully installed PyMGA-0.0.1 
```

The package is now available for system wide use

###### Install with python

Execute the following command from the package folder, e.g. `src/PyMGA`

```
src/PyMGA> python setup.py install
```


## PyMGA.methods

#### PyMGA.methods.MGA(case)

**PyMGA.methods.MGA.find_optimum()**
Finds the cost optimal solution of the case object given

**PyMGA.methods.MGA.serach_directions(n_samples, n_workers)**
Performs the MGA study on the case study. The method draws random search directions uniformly over the hypersphere.  

*n_samples:* The number of samples to draw  
*n_workers:* number of parallel process to start. Default=4

#### PyMGA.methods.MAA

**PyMGA.methods.MAA.find_optimum()**
Finds the cost optimal solution of the case object given

**PyMGA.method.MAA.search_directions(self, n_samples, n_workers, max_iter)**

Runs the MAA algorithm documented in [Modeling all alternative solutions for highly renewable energy systems](https://doi.org/10.1016/j.energy.2021.121294)

*n_samples:* Maximum number of samples to draw  
*n_workers:* number of parallel process to start. Default=4  
*max_iter:* Maximum number of MAA iterations  


#### PyMGA.methods.rMAA

**PyMGA.methods.rMAA.find_optimum()**
Finds the cost optimal solution of the case object given

**PyMGA.methods.rMAA.serach_directions(n_samples, har_samples, n_workers, max_iter, tol)**

*n_samples:* Maximum number of samples to draw  
*har_samples:* Number of MAA samples to draw when computing acceptance rate and finding new directions. Default=5000  
*n_workers:* number of parallel process to start. Default=4  
*max_iter:* maximum number of iterations to perfom. Default = 30  
*tol:* The acceptance rate required before terminating, unless n_samples is reached first. A number between 0-1. Default = 0.99  

#### PyMGA.cases

**PyMGA.cases.PyPSA_case(config, base_network_path variables=None, tmp_network_path='tmp/networks/tmp.h5', n_snapshots=100, mga_slack=0.1)**

This class represents a PyPSA_Eur or PyPSA_Eur_Sec network as a case that can be investigated by the MGA methods. 

*config:* Config file containing the solver options
*base_network_path:* Path to the network file 
*variables:* A dict specifying the variables to be investigated by the MGA/MAA methods 
*n_snapshots:* Number of snapshots to include from the given network. 
*mga_slack:* Percent MGA slack to use as a fraction, e.g. 0.1 = 10%
*tmp_network_path:* Path where to store temporary network files


**PyMGA.cases.Cube(dim,cuts)**
A synthetic tescase of testing MGA/MAA methods. The method creates an optimization problem with a solution space in the form of a cube sliced with n cuts. 

*dim:* Number of dimensions of the test case
*cuts:* Number of cuts

**PyMGA.cases.CubeCorr(dim)**
A synthetic tescase of testing MGA/MAA methods. The method creates an optimization problem with a solution space in the form of a cube sliced by parallel planes to give the space strong correlations between variables.

*dim:* Number of dimensions of the test case

**PyMGA.cases.CrossPoly(dim)**
A synthetic tescase of testing MGA/MAA methods. The method creates an optimization problem with a solution space in the form of the intersection of a hyperube and a cross-polytope.

*dim:* Number of dimensions of the test case
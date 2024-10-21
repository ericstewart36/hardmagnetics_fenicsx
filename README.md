# Magneto-viscoelastic snap-through modeling codebase

This repository contains the code base for the FEniCSx finite-element implementation of the large deformation magneto-viscoelasticity theory I developed for modeling snap-through processes in bistable hard-magnetic soft-elastomeric (or hard-magnetorheological "h-MRE") actuators. 

The codes in this repository are updated versions of the original "legacy" FEniCS hard-magnetics codes published in [this repository](https://github.com/ericstewart36/hardmagnetics/). The updated codes: 
- Are written in the modern FEniCSx version of FEniCS platform, using Jupyter notebooks.
- Use an updated theoretical formulation of finite viscoelasticity and its numerical implementation in FEniCSx following [Stewart and Anand (2024)](https://doi.org/10.1016/j.ijsolstr.2024.113023). 

This repository contains all the relevant FEniCSx Jupyter notebooks, mesh files, and experimental data files which were used in the calibration, prediction, and demonstration studies in [Stewart and Anand (2023)](https://doi.org/10.1016/j.jmps.2023.105366), as well as three additional simulations regarding the quasi-static bending behavior of h-MRE beams. 

# Running the codes

A detailed guide for installing FEniCSx in a Docker container and running the notebooks using VSCode is provided in this repository, both for [Mac](https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FEniCSx_v08_Docker_install_mac.pdf) and [Windows](https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FEniCSx_v08_Docker_install_windows.pdf). The installation process is essentially similar for the two operating systems, but the example screenshots in the instructions are from the relevant system.

These are our preferred methods for editing and running FEniCSx codes, although [many other options exist](https://fenicsproject.org/download/). Note that all codes were written for FEniCSx v0.8.0, so our instructions documents will direct you to install this specific version of FEniCSx.

We have also provided a python script version of simulation HM08 which is meant to be run with MPI parallelization. To run this script in parallel on e.g. four cores use the following command syntax in the terminal:  

```
mpirun -n 4 python3 HM08_hemisphere_eversion.ipynb
```

![](https://github.com/ericstewart36/hardmagnetics/blob/main/example_animation.gif)

# Citations

- E. M. Stewart and L. Anand. Magneto-viscoelasticity of hard-magnetic soft-elastomers: Application to modeling the dynamic snap-through behavior of a bistable arch. *Journal of the Mechanics and Physics of Solids*, 179:105336, Oct. 2023.

- E. M. Stewart and L. Anand. A large deformation viscoelasticity theory for elastomeric materials and its numerical implementation in the open-source finite element program FEniCSx. *International Journal of Solids and Structures*, 303:113023, Oct. 2024.

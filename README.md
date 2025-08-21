### PermMag

This repository contains Python code for simulating and visualising permanent magnet configurations using `Magpylib`. The initial setup generates a quadrupole field which is required for a 2D magneto optical trap. 

The code allows users to program magnets in specific locations to generate custom fields, where the corresponding vector field plots and field component plots are generated. 

This code was initially written to try and solve the issue of the radial gradient value dropping off as you move further away from the centre of the trap in the axial direction. 

In short, the code will:

- Construct the magnet assembly

- Show a 3D visualisation of the magnets

- Plot the magnetic field components in the radial and axial directions

- Calculate and display field gradients in Gauss/cm


This code was written as part of a summer project within the EQOP research group at the University of Strathclyde. I received funding for this project from the Carnegie Trust under their Undergraduate Vacation Scholarship scheme, for which I am grateful.  
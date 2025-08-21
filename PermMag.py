#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 11:09:22 2025

@author: Sam
"""

import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
import plotly.io as pio
from scipy.spatial.transform import Rotation as R

pio.renderers.default = "browser"

# =============================================================================
# Creating permanent bar magnet system for quadrupole field 
# =============================================================================

# Geometry of magnet positions within each corner (3 magnet locations)
magnet_offsets = [
    (19e-3, 19e-3),
    (18.293e-3, 19.707e-3),
    (19.707e-3, 18.293e-3),
]

# Magnet dimensions and strength
x_l, y_l, z_l = 1.5e-3, 1e-3, 5e-3
M = 1074295  # N45 Neodymium

# Z offset per layer
z_offset = 5e-3  # 5 mm

# Corner rotations
rot_top_right   = R.from_euler('z',  45, degrees=True)
rot_top_left    = R.from_euler('z', 315, degrees=True)
rot_bottom_left = R.from_euler('z',  45, degrees=True)
rot_bottom_right= R.from_euler('z', 315, degrees=True)

# =============================================================================
# Define layer configurations for each magnet in each corner
# Each corner contains 3 magnet positions, and each list defines
# the specific z-layers (indices) at which magnets are placed
# =============================================================================
heights = {
    "C1": [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [], []],  # top right
    "C2": [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [], []],  # top left
    "C3": [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [], []],  # bottom left
    "C4": [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [], []],  # bottom right
}

# =============================================================================
# Utility function to build an individual stack at a given (x, y)
# and for specific z-layers
# =============================================================================
def build_custom_stack(x, y, sign_x, sign_y, rot, M_sign, z_layers):
    stack = magpy.Collection()
    for z_index in z_layers:
        m = magpy.magnet.Cuboid(
            dimension=(x_l, y_l, z_l),
            magnetization=(0, M_sign * M, 0),
            position=(sign_x * x, sign_y * y, z_index * z_offset)
        )
        m.rotate(rot, anchor=None)
        stack.add(m)
    return stack

# =============================================================================
# Function to build a full corner from 3 stacks
# =============================================================================
def build_corner(z_layer_lists, sign_x, sign_y, rot, M_sign):
    corner = magpy.Collection()
    for (x, y), z_layers in zip(magnet_offsets, z_layer_lists):
        stack = build_custom_stack(x, y, sign_x, sign_y, rot, M_sign, z_layers)
        corner.add(stack)
    return corner

# Build all four corners
Corner_1 = build_corner(heights["C1"], +1, +1, rot_top_right,   +1)
Corner_2 = build_corner(heights["C2"], -1, +1, rot_top_left,    +1)
Corner_3 = build_corner(heights["C3"], -1, -1, rot_bottom_left, -1)
Corner_4 = build_corner(heights["C4"], +1, -1, rot_bottom_right,-1)

# Combine all corners into one system
Full_Assembly = magpy.Collection(Corner_1, Corner_2, Corner_3, Corner_4)

# Visualize
Full_Assembly.show()

# =============================================================================
# Calculating and plotting field components
# =============================================================================

# =============================================================================
# # =============================================================================
# # Vector plot for Magnetic Field in the x–y plane 
# # =============================================================================
# 
# #Creating grid which will assign component to points in space
# xy_ts = np.linspace(-200e-3, 200e-3, 100)
# xy_grid = np.array([[(x, y, 0) for x in xy_ts] for y in xy_ts])  # z = 0 plane
# X_xy, Y_xy, _ = np.moveaxis(xy_grid, 2, 0)
# 
# B_field_xy = Stacked_System.getB(xy_grid)
# B_x_xy, B_y_xy, _ = np.moveaxis(B_field_xy, 2, 0)
# 
# #Computing magnitude
# B_radial_xy = np.sqrt(B_x_xy**2 + B_y_xy**2)
# B_radial_xy /= np.max(B_radial_xy) + 1e-12  # normalize for color map
# 
# #Plotting vector field
# fig, ax = plt.subplots(figsize=(8, 6))
# stream = ax.streamplot(X_xy, Y_xy, B_x_xy, B_y_xy, color=B_radial_xy, linewidth=1.5, density=2, cmap='plasma')
# ax.set_aspect('equal')
# ax.set_title('Magnetic Field in x–y Plane')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.colorbar(stream.lines, ax=ax, label='Normalized Field Strength')
# plt.tight_layout()
# plt.show()
# 
# =============================================================================
# =============================================================================
# Plotting X component of B with y = 0 and z = 0
# =============================================================================

#Creating line for measuremtn of B components along x 
x_ts = np.linspace(-100e-3, 100e-3, 100)
x_measurement_line = np.array([[x, 0, 50e-3] for x in x_ts])   ###!!! Z needs to be 5e-3 if you have a stack of two on top pf each pther, this changes the measurment line to the correct psoition to preserve symetry
#47.5e-3 for 20 stack of these magnets
#x,y and z B components along the x axis
B_forgraph_x = Full_Assembly.getB(x_measurement_line)

#Spliced array so it contains just the x components; 100, 1 dimension array
B_x_spliced_in_x = B_forgraph_x[:, 0]
#B_amp_z = np.linalg.norm(B_z_spliced, axis=0)

#Spliced array so it contains just the y components; 100, 1 dimension array
B_y_spliced_in_x = B_forgraph_x[:, 1]

#Spliced array so it contains just the z components; 100, 1 dimension array
B_z_spliced_in_x = B_forgraph_x[:, 2]

#Creating Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8) , sharex=True)
ax.plot(x_ts, B_x_spliced_in_x, label = 'Bx in x', color = 'r')
ax.plot(x_ts, B_y_spliced_in_x, label = 'By in x ', color = 'g')
ax.plot(x_ts, -B_z_spliced_in_x, label = 'Bz in x', color = 'b')

ax.set_ylabel('B (T)')
ax.set_xlabel('x (mm)')
ax.set_title('B Field Components in x direction')
ax.grid(True)
ax.legend()


# =============================================================================
# Plotting z component of B with x = 0 and y = 0
# =============================================================================

#Creating line for measurement for B components along z axis
z_ts = np.linspace(-100e-3, 100e-3, 100)
z_measurement_line = np.array([[0, 0, z] for z in z_ts]) 

#Calculating x, y and z components of B field along z axis
B_forgraph_z = Full_Assembly.getB(z_measurement_line)

#Spliced array so it contains just the x components; 100, 1 dimension array
B_x_spliced_in_z = B_forgraph_z[:, 0]

#Spliced array so it contains just the y components; 100, 1 dimension array
B_y_spliced_in_z = B_forgraph_z[:, 1]

#Spliced array so it contains just the z components; 100, 1 dimension array
B_z_spliced_in_z = B_forgraph_z[:, 2]

#Creating Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8) , sharex=True)
ax.plot(z_ts, B_x_spliced_in_z, label = 'Bx in z', color = 'r')
ax.plot(z_ts, B_y_spliced_in_z, label = 'By in z ', color = 'g')
ax.plot(z_ts, B_z_spliced_in_z, label = 'Bz in z', color = 'b')

ax.set_ylabel('B (T)')
ax.set_xlabel('z (m)')
ax.set_title('B Field Components in z direction')
ax.grid(True)
ax.legend()

# =============================================================================
# # =============================================================================
# # Vector plot for Magnetic Field in the x–z plane 
# # =============================================================================
# 
# #Creating grid which will assign component to points in space
# xz_ts = np.linspace(-100e-3, 100e-3, 100)
# xz_grid = np.array([[(x, 0, z) for x in xz_ts] for z in xz_ts])  # z = 0 plane
# X_xz, _, Z_xz = np.moveaxis(xz_grid, 2, 0)
# 
# B_field_xz = Stacked_System.getB(xz_grid)
# B_x_xz, _, B_z_xz = np.moveaxis(B_field_xz, 2, 0)
# 
# #Computing magnitude
# B_radial_xz = np.sqrt(B_x_xz**2 + B_z_xz**2)
# B_radial_xz /= np.max(B_radial_xz) + 1e-12  # normalize for colour map
# 
# #Plotting vector field
# fig, ax = plt.subplots(figsize=(8, 6))
# stream = ax.streamplot(X_xz, Z_xz, B_x_xz, B_z_xz, color=B_radial_xz, linewidth=1.5, density=2, cmap='plasma')
# ax.set_aspect('equal')
# ax.set_title('Magnetic Field in x–z Plane')
# ax.set_xlabel('x')
# ax.set_ylabel('z')
# plt.colorbar(stream.lines, ax=ax, label='Normalized Field Strength')
# plt.tight_layout()
# plt.show()
# =============================================================================

# =============================================================================
# Calculating and graphing the gradiens of the B Components
# =============================================================================

# =============================================================================
# #Gradient of B in pos x direction
# =============================================================================

x_ts_Grad = np.linspace(-0.0025, 0.0025, 100)
dx = x_ts[99] - x_ts[98]


#Calculating gradient of Bx in x, y and z
Grad_Bx_in_x = np.gradient(B_x_spliced_in_x, dx )

Grad_By_in_x = np.gradient(B_y_spliced_in_x, dx)
Grad_Bz_in_x = np.gradient(B_z_spliced_in_x, dx)


#Plotting these along x linspace array
fig, ax = plt.subplots(1, 1, figsize=(10, 8) , sharex=True)
ax.plot(x_ts, Grad_Bx_in_x, label = 'Grad Bx in x', color = 'r')
ax.plot(x_ts, Grad_By_in_x, label = 'Grad By in x', color = 'g')
ax.plot(x_ts, Grad_Bz_in_x, label = 'Grad Bz in x', color = 'b')

ax.set_ylabel('B (T)')
ax.set_xlabel('x (m)')
ax.set_title('Grad of B Field Components in x direction')
ax.grid(True)
ax.legend()

# =============================================================================
# Gradient of B in pos z direction 
# =============================================================================

dz = z_ts[99]-z_ts[98]

#Calculating gradient of Bz in x, y and z direction
Grad_Bx_in_z = np.gradient(B_x_spliced_in_z, dz)
Grad_By_in_z = np.gradient(B_y_spliced_in_z, dz)
Grad_Bz_in_z = np.gradient(B_z_spliced_in_z, dz)

#Plotting aginst z linspace array
fig, ax = plt.subplots(1, 1, figsize=(10, 8) , sharex=True)
ax.plot(z_ts, Grad_Bx_in_z, label = 'Grad Bx in z', color = 'r')
ax.plot(z_ts, Grad_By_in_z, label = 'Grad By in z', color = 'g')
ax.plot(z_ts, Grad_Bz_in_z, label = 'Grad Bz in z', color = 'b')

ax.set_ylabel('B (T)')
ax.set_xlabel('z (mm)')
ax.set_title('Grad of B Field Components in z direction')
ax.grid(True)
ax.legend()

# Find max gradient in T/mm
Max_Grad_Bx_in_x = np.max(Grad_Bx_in_x)
Max_Grad_Bz_in_z = np.max(Grad_Bz_in_z)
Max_Grad_Bx_in_z = np.max(Grad_Bx_in_z)
Max_Grad_Bz_in_x = np.max(Grad_Bz_in_x)

# Convert to G/cm: (T → G is x10⁴, mm → cm is ÷0.1 → x10)
#max_gradient_G_per_cm = Max_Grad_Bx_in_x * 1e4 * 10

# Print result
#print(f"Maximum gradient: {Max_Grad_Bx_in_x:.6e} T/mm")
#print(f"Converted to: {max_gradient_G_per_cm:.2f} G/cm")

# Value in Tesla per meter
gradient_T_per_m = Max_Grad_Bx_in_x
Gradient_of_Bz_inz_inT_m = Max_Grad_Bz_in_z
Gradient_of_Bx_inz_inT_m = Max_Grad_Bx_in_z
Gradient_of_Bz_inx_inT_m = Max_Grad_Bz_in_x

# Conversion factor: 1 T/m = 100 G/cm
conversion_factor = 1e4/100

# Convert to Gauss per centimeter
gradient_Bx_inx_G_per_cm = gradient_T_per_m * conversion_factor
gradient_Bz_inz_G_per_cm = Gradient_of_Bz_inz_inT_m * conversion_factor
gradient_Bx_inz_G_per_cm = Gradient_of_Bx_inz_inT_m * conversion_factor
gradient_Bz_inx_G_per_cm = Gradient_of_Bz_inx_inT_m * conversion_factor 


# Print the result
print(f"Magnetic field gradient of Bx in x: {gradient_Bx_inx_G_per_cm:.4f} G/cm")
print(f"Magnetic Field Gradient of Bz in z: {gradient_Bz_inz_G_per_cm:.4f} G/cm")
print(f"Magnetic Field Gradient of Bx in z: {gradient_Bx_inz_G_per_cm:.4f} G/cm")
print(f"Magnetic Field Gradient of Bz in x: {gradient_Bz_inx_G_per_cm:.4f} G/cm")




import magpylib as mag

# Define x-axis sampling
x_range = np.linspace(-0.1, 0.1, 100)  # 10 cm span along x
dx = x_range[99] - x_range[98]        # spacing for np.gradient
y_val = 0                              # fixed y

# Define z values to sample (from -10 mm to 100 mm in 5 mm steps)
z_vals = np.arange(-0.01, 0.100, 0.005)  # in meters

# Store max gradients for each height
max_gradients = []

for z in z_vals:
    # Define positions along x at fixed y and current z
    positions = np.array([[x, y_val, z] for x in x_range])
    
    # Get B field at each point
    B = mag.getB(Full_Assembly, positions)
    Bx = B[:, 0]  # x-component of B field

    # Compute gradient of Bx with respect to x
    dBxdx = np.gradient(Bx, dx) * 100  # convert to cm⁻¹ if needed

    # Append the maximum value of the gradient array
    max_gradients.append(np.max(dBxdx))

# Plot maximum gradient vs height z
plt.figure(figsize=(8, 5))
plt.plot(z_vals * 1000, max_gradients, marker='o', color='blue')
plt.xlabel("Height along Radial Axis (mm)")
plt.ylabel("Max Gradient of Radial Component")

# Optional vertical grid lines every 5 mm from -5 mm to 105 mm
for mm in range(-5, 110, 5):
    plt.axvline(x=mm, color='k', linestyle='--', linewidth=1)

plt.grid(True)
plt.tight_layout()
plt.show()



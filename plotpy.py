import numpy as np
import matplotlib.pyplot as plt

filename = 'MYVelfloris_0_011.0_3d.npy'
data = np.load(filename)
print(data.shape)

var_names = [ 'Vel_z', 'X', 'Y', 'Z']
vel_z_idx = var_names.index('Vel_z')
X_idx = var_names.index('X')
Y_idx = var_names.index('Y')
Z_idx = var_names.index('Z')


vel_z = data[vel_z_idx]
x=data[X_idx]
y=data[Y_idx]
z=data[Z_idx]
print(vel_z.shape)
print(x.shape)
print(y.shape)
print(z.shape)
#Z = data[Z_idx]

slice_index = 13
#slice_data = np.squeeze(vel_z[slice_index, :, :])
slice_data = np.squeeze(data[slice_index, :, :])


# Plot the 2D slice of 'Vel_x'
plt.figure()
plt.imshow(slice_data, aspect='auto', cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Slice of Vel_x')

#plt.clim(-1, 1)
plt.show()# -*- coding: utf-8 -*-


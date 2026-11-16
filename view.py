# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-03-18 21:32:20
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-06-11 23:15:03
import napari
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#P = np.load("ct_result_stress_vm.npy")
data = np.load('ct_result_results.npy')
print(data.shape)

# load multichannel image in one line
viewer, image_layers = napari.imshow(data, channel_axis=3,
                       name=["stress", "strain"],
                       colormap=["turbo", "turbo"])

for layer in image_layers:
    layer.colorbar.visible = True
    layer.visible = False
image_layers[1].visible = True


# Calculate contrast limits centered at median for better diverging visualization
#median = np.median(P)
#max_val = np.max(P)
#min_val = np.median(P) - (max_val - np.median(P))
#min_val = max(0, min_val)
#min_val = 0.0
#max_val = 0.04

#viewer, image_layer = napari.imshow(P, colormap='turbo', contrast_limits=[min_val, max_val])
#image_layer.colorbar.visible = True
#viewer.dims.order = (1, 2, 0)


if __name__ == '__main__':
    napari.run()
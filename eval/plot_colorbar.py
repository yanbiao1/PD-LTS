
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import ImageColor
from matplotlib import colors
import numpy as np

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

vals = np.ones((256, 4))


# Light blue to Dark blue
# min_color = np.array([213, 225, 230]) / 256  # np.array([118, 170, 232]) / 256
# max_color = np.array([44, 52, 64])    / 256  # np.array([255, 55, 55]) / 256
# vals[:, 0] = np.linspace(min_color[0], max_color[0], 256)
# vals[:, 1] = np.linspace(min_color[1], max_color[1], 256)
# vals[:, 2] = np.linspace(min_color[2], max_color[2], 256)

# orange to red
# min_color = np.array([255, 239, 204]) / 256
# mid_color = np.array([255, 153, 102]) / 256
# max_color = np.array([255, 0, 0]) / 256

# Purpose-blue
min_color = np.array([230, 230, 255]) / 256
mid_color = np.array([128, 128, 255]) / 256
max_color = np.array([0, 0, 153]) / 256

# Green
# min_color = np.array(list(ImageColor.getcolor("#F3FAEE", "RGB"))) / 256
# mid_color = np.array(list(ImageColor.getcolor("#3A631C", "RGB"))) / 256
# max_color = np.array(list(ImageColor.getcolor("#1C3609", "RGB"))) / 256

# orange
# min_color = np.array(list(ImageColor.getcolor("#FFE8DC", "RGB"))) / 256
# mid_color = np.array(list(ImageColor.getcolor("#FC600A", "RGB"))) / 256
# max_color = np.array(list(ImageColor.getcolor("#341809", "RGB"))) / 256

# blue
# min_color = np.array(list(ImageColor.getcolor("#E6ECFE", "RGB"))) / 256
# mid_color = np.array(list(ImageColor.getcolor("#7195F9", "RGB"))) / 256
# max_color = np.array(list(ImageColor.getcolor("#0F4CF5", "RGB"))) / 256

# blue2
# min_color = np.array([239, 244, 254]) / 256
# mid_color = np.array([31, 139, 235]) / 256
# max_color = np.array([38, 88, 148]) / 256

vals[:, 0] = np.concatenate([np.linspace(min_color[0], mid_color[0], 127), np.linspace(mid_color[0], max_color[0], 129)])
vals[:, 1] = np.concatenate([np.linspace(min_color[1], mid_color[1], 127), np.linspace(mid_color[1], max_color[1], 129)])
vals[:, 2] = np.concatenate([np.linspace(min_color[2], mid_color[2], 127), np.linspace(mid_color[2], max_color[2], 129)])


color_cmp = colors.ListedColormap(vals)

# cmap = mpl.cm.cool
cmap = color_cmp
# norm = mpl.colors.Normalize(vmin=0.0, vmax=0.025)
norm = mpl.colors.Normalize()

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal', ticks=[])
# cbar.ax.set_xticklabels(['0.0', '0.00125', '0.025'])

# cbar.set_ticks([0.0, 1.0])
# cbar.set_ticklabels(['Noisy', 'Clean'])

# cbar.ax.set_xlabel('Noisy', loc='left')
# cbar.ax.set_ylabel('Clean', loc='bottom')

plt.show()

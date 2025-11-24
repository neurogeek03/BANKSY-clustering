import pickle 
import os
from banksy.plot_banksy import plot_results
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Generate 30 visually distinct colors with Seaborn
colors = sns.color_palette("tab20", 30)  # RGB tuples in 0–1 range

# Convert to a Matplotlib colormap
c_map = ListedColormap(colors)

# # 30 visually distinct colors (RGB tuples in 0–1 range)
# colors_30 = [
#     (0.894, 0.102, 0.110), (0.215, 0.494, 0.721), (0.302, 0.686, 0.290),
#     (0.596, 0.306, 0.639), (1.000, 0.498, 0.000), (1.000, 1.000, 0.200),
#     (0.651, 0.337, 0.157), (0.969, 0.506, 0.749), (0.600, 0.600, 0.600),
#     (0.894, 0.216, 0.518), (0.102, 0.596, 0.894), (0.706, 0.894, 0.102),
#     (0.337, 0.651, 0.157), (0.498, 0.000, 1.000), (0.980, 0.690, 0.216),
#     (0.310, 0.180, 0.600), (0.800, 0.102, 0.294), (0.216, 0.490, 0.976),
#     (0.400, 0.800, 0.400), (0.600, 0.400, 0.800), (1.000, 0.600, 0.000),
#     (0.976, 0.980, 0.310), (0.400, 0.600, 0.200), (0.690, 0.216, 0.980),
#     (0.894, 0.310, 0.180), (0.102, 0.894, 0.216), (0.310, 0.400, 0.800),
#     (0.800, 0.400, 0.600), (0.216, 0.980, 0.690), (0.600, 0.200, 0.400),
#     (0.980, 0.310, 0.400)
# ]

# # Convert the list to a Matplotlib colormap
# c_map = ListedColormap(colors_30)

project_dir = '/scratch/mfafouti/BANKSY'
file_path = os.path.join(project_dir, 'out')

coord_keys = ('xcoord', 'ycoord', 'coord_xy')
coord_keys = list(coord_keys)
coord_keys[2] = 'X_spatial'

with open(os.path.join(project_dir, "banksy_dict.pkl"), "rb") as f:
    banksy_dict = pickle.load(f)

with open(os.path.join(project_dir, "results_df.pkl"), "rb") as f:
    results_df = pickle.load(f)

with open(os.path.join(project_dir, "max_num_labels.pkl"), "rb") as f:
    max_num_labels = pickle.load(f)

# c_map = "rainbow"
weights_graph =  banksy_dict['scaled_gaussian']['weights'][0]

# plot_results(
#     results_df,
#     weights_graph,
#     c_map=c_map,
#     match_labels = True,
#     coord_keys = coord_keys,
#     max_num_labels  =  max_num_labels, 
#     save_path = os.path.join(file_path, 'tmp_png'),
#     save_fig = True
# )

plot_results(
    results_df,
    weights_graph,
    c_map=c_map,
    match_labels=True,
    coord_keys=coord_keys,
    max_num_labels=max_num_labels,
    save_path=os.path.join(file_path, 'tmp_png'),
    save_fig=True,
    save_fullfig=True,        # ← this saves the entire composite figure
    plot_heat_map=False,      # optional
    plot_dot_plot=False       # optional
)

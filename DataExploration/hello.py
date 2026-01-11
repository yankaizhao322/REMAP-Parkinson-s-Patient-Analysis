import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Set the matplotlib backend to 'Agg' to avoid GUI issues
matplotlib.use('Agg')

# Define the correct path to the STS 2D skeleton data
file_path = '/Users/kaiyangqian/Downloads/21h9f9e30v9cl2fapjggz4q1x7/SitToStand/Data/STS_2D_skeletons_coarsened/Pt204_C_n_301.csv'

# Load the CSV file and check its shape
keypoints_csv = np.loadtxt(file_path, delimiter=',')
print("Original CSV Shape:", keypoints_csv.shape)

# Extract only the necessary columns (50 columns for 25 joints with x,y coordinates)
# Skipping the first column (frame number) and the second column (time)
keypoints_data = keypoints_csv[:, 2:52]
print("Filtered Data Shape:", keypoints_data.shape)

# Reshape the data correctly
number_of_joints = 25
dimension = 2
keypoints = keypoints_data.reshape(keypoints_data.shape[0], number_of_joints, dimension)
print("Reshaped Keypoints Shape:", keypoints.shape)

# Define joint connections for the 25-joint STS data
connections = [(17, 15), (15, 0), (0, 16), (16, 18), (0, 1), (1, 5), (5, 6), (6, 7), (1, 2), (2, 3), (3, 4),
               (1, 8), (8, 12), (12, 13), (13, 14), (14, 19), (19, 20), (14, 21), (8, 9), (9, 10), (10, 11),
               (11, 22), (22, 23), (11, 24)]

# Define colors for visualization
lcolor = (255, 0, 0)  # Red for left side
rcolor = (0, 0, 255)  # Blue for right side
joint_color = (0, 255, 0)  # Green for joints
base_thickness = 1

# Loop through each frame to visualize the skeleton
for idx, kps in enumerate(keypoints):
    clear_output(wait=True)

    max_dim = round(np.max(kps) * 1.3)
    img = np.zeros((max_dim, max_dim, 3), dtype='uint8')

    for i, c in enumerate(connections):
        start = tuple(map(int, kps[c[0]]))
        end = tuple(map(int, kps[c[1]]))

        # Determine the thickness of the lines
        thickness = 2 if i in [8] else base_thickness

        # Draw lines and joints
        cv2.line(img, start, end, lcolor if i % 2 == 0 else rcolor, thickness)
        cv2.circle(img, start, radius=3, color=joint_color, thickness=-1)
        cv2.circle(img, end, radius=3, color=joint_color, thickness=-1)

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(img)
    plt.title(f'Frame {idx + 1}/{len(keypoints)}')

    plt.rcParams['figure.dpi'] = 120
    plt.show()
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')


import cv2
import bson
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt, gridspec
# from image_utils import b64_image_to_numpy, crop_resize_image
import image_utils
from statsmodels.formula.api import ols
from scipy.signal import medfilt
from scipy.stats import pearsonr, spearmanr, ttest_ind, wilcoxon
# from sphere_utils import *

# Define rdsf_path to your local REMAP dataset directory
rdsf_path = Path('/Users/kaiyangqian/Downloads/21h9f9e30v9cl2fapjggz4q1x7')

# Verify if the path exists
if not rdsf_path.exists():
    raise FileNotFoundError(f"Directory not found: {rdsf_path}")

print(f"rdsf_path is set to: {rdsf_path}")

def skeleton_sizes(pose):
    this_skel = pose.copy()
    bf = this_skel[:-1].reshape((-1, 2))
    mk = np.any(bf == 0, axis=1)[:, None]
    bf[np.hstack((mk, mk))] = np.nan
    width = np.nanmax(bf[:, 0]) - np.nanmin(bf[:, 0])
    height = np.nanmax(bf[:, 1]) - np.nanmin(bf[:, 1])
    return width, height

def plot_skeleton(pose, **kwargs):
    adjs = np.array([(17, 15), (15, 0), (0, 16), (16, 18), (0, 1), (1, 5), (5, 6), (6, 7), (1, 2), (2, 3), (3, 4),
            (1, 8), (8, 12), (12, 13), (13, 14), (14, 19), (19, 20), (14, 21), (8, 9), (9, 10), (10, 11),
            (11, 22), (22, 23), (11, 24)])
    pose2 = pose.copy()
    pose2[pose2 == 0] = np.nan
    plt.plot([pose2[adjs[:, 0] * 2 + 0], pose2[adjs[:, 1] * 2 + 0]],
             [pose2[adjs[:, 0] * 2 + 1], pose2[adjs[:, 1] * 2 + 1]], **kwargs)

# Paths
data_folder = r'..\..\Data\STS_2D3D_skeletons'
# labels_path = r'\\rdsfcifs.acrc.bris.ac.uk\Restricted_PD_SENSORS\dataset_publication\SitToStand_12.02.2023_ExtraLabels.xls'
# Update the labels_path to your local dataset path
labels_path = r'/Users/kaiyangqian/Downloads/21h9f9e30v9cl2fapjggz4q1x7/SitToStand/Data/STS_human_labels/SitToStand_human_labels.xls'

data_folder = Path(data_folder)
labels_path = Path(labels_path)

fps = 30.0

# Read the labels file
df = pd.read_excel(labels_path)

# Best params for dur
medfilt_k = 25
min_mask = 0.8486
smooth_kernel = 11
smooth_poly = 1
zero_cross_thr = 0.94
zz_which_joints = 4


sts_files = [f for f in data_folder.glob('*.csv')]
for sts_file in sts_files:
    part_id = sts_file.name.split('_')[0]
    pd_or_c = sts_file.name.split('_')[1]
    excel_id = sts_file.name.split('_')[3]

    data = np.loadtxt(sts_file, delimiter=',', skiprows=1)
    skeletons = data[:, 1:]
    frames = data[:, 0]

    # Process the bounding box data
    # data = matching_data[id_target]
    # frame = matching_frame[id_target]

    # Remove zeros
        # hips: 9 8 12
    # head: 0 15 16 17 18
    # shoulders: 2 1 5
    # head_joints = [0, 15, 16, 17, 18, 2, 1, 5]
    # head_joints = [0, 15, 16, 17, 18, 2, 1, 5]
    if zz_which_joints == 0:
        head_joints = [0, 15, 16, 17, 18, 2, 1, 5, 9, 8, 12]
    elif zz_which_joints == 1:
        head_joints = [0, 15, 16, 17, 18, 2, 1, 5]
    elif zz_which_joints == 2:
        head_joints = [0, 15, 16, 17, 18]
    elif zz_which_joints == 3:
        head_joints = [0]
    elif zz_which_joints == 4:
        head_joints = [2, 1, 5]
    elif zz_which_joints == 5:
        head_joints = [9, 8, 12]
    # head_joints = [0, 15, 16, 17, 18]
    all_masks = []
    all_joints = []
    stop = False
    non_valid = []
    for head_joint in head_joints:
        y_joint = data[:, head_joint * 2 + 1]
        mask = y_joint == 0
        non_valid.append(np.sum(mask))

    best_head = np.argmin(non_valid)
    head_joint = head_joints[best_head]

    y_joint = data[:, head_joint * 2 + 1]
    mask = y_joint == 0
    if mask.sum() / len(mask) > min_mask or len(mask) < 11:  # 0.4:
        # print(f'#{index} - Invalid/valid poses is {mask.sum() / len(mask) * 100:.1f}%. Nothing to do here')
        continue

    # Replace missing values
    y_joint[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_joint[~mask])

    # mask = np.any(np.vstack(all_masks), axis=0)
    # if mask.sum()/len(mask) > 0.1: #0.4:
    #     print(f'#{index} - Invalid/valid poses is {mask.sum()/len(mask)*100:.1f}%. Nothing to do here')
    #     continue

    y_head = y_joint.copy()

    # y_head = data[:, 1]
    # # y_head = (data[:, 9*2+1] + data[:, 8*2+1] + data[:, 12*2+1])/3
    # y_stern = data[:, 3]
    # mask = np.any(np.vstack((y_head==0, y_stern==0)), axis=0)#y_head==0
    # if mask.sum()/len(mask) > 0.1: #0.4:
    #     print(f'#{index} - Invalid/valid poses is {mask.sum()/len(mask)*100:.1f}%. Nothing to do here')
    #     continue
    #
    # if len(y_head) < 11:
    #     print(f'#{index} - Skipping because not enough data points: {len(y_head)}')
    #     continue
    # y_head[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_head[~mask])
    # y_stern[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_stern[~mask])
    # # np.where(y_head == 0)
    # # plt.figure()
    # # plt.plot(y_head_sts)
    #
    # # Calculate the calibration
    # mean_neck_height = np.mean(np.abs(y_stern - y_head))
    # # mean_neck_height2 = np.median(np.abs(y_stern - y_head))

    # Calculate the STS speed
    time = frames / fps
    if np.any(np.diff(time) > 1 / fps * 1.1):
        print(f'{participant_id} - #{index} - Skipping because of non-contiguous clip: {np.diff(time).max():.1f} sec')
        continue
    cy = -y_head  # Use the y coordinate of the upper edge of the 3D bounding box
    # cy = medfilt(cy, kernel_size=3)
    cy = medfilt(cy, kernel_size=medfilt_k)
    # speed, duration, snr, extent, time_i = speed_of_ascent(cy, time, direction, debug=debug_sts, smooth_kernel=25, smooth_poly=3)
    speed, duration, snr, extent, time_i, time_end = speed_of_ascent(cy, time, 'stand up', debug=False,
                                                                     smooth_kernel=smooth_kernel,
                                                                     smooth_poly=smooth_poly, zct=zero_cross_thr)
    # w, h = skeleton_sizes(data[sts_slice, :][time_end, :])
    # if time_end > 0:
    #     h = np.mean(np.array(list(map(skeleton_sizes, data[sts_slice, :][time_i:time_end]))), axis=0)[1]
    # else:
    w, h = skeleton_sizes(data[:, :][time_end, :])
    df.at[index, 'sts_speed'] = speed / h * 1.7  # m/s
    df.at[index, 'subj_height'] = h
    df.at[index, 'sts_duration'] = duration
    df.at[index, 'snr'] = snr
    df.at[index, 'extent'] = extent
    # df.at[index, 'neck_height'] = np.abs(y_stern[time_i] - y_head[time_i])
    # df.at[index, 'neck_height2'] = mean_neck_height





















# Add new column for sit-to-stand speed
df['sts_speed'] = None
df['sts_duration'] = None
df['snr'] = None
df['extent'] = None
df['subj_height'] = None
df['neck_height'] = None
df['neck_height2'] = None
df = df.reset_index()  # make sure indexes pair with number of rows

# Participants folder
participants = [bf for bf in rdsf_path.glob('sub-*')]

uid = 0
# skip = 7
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Find bounding box for this entry
    participant_id = row['Participant ID number']
    sts_timestamp = row['Timestamp start of sts_final_attempt_duration']

    if row['Transition ID'] in sts_skip:
        continue

    if pd.isna(row['NUC file names']) or row['NUC file names'] == '-':
        print(f"{participant_id} - Skipping row {index} because empty file name label")
        continue
    candidate_folders = row['NUC file names'].split(',')

    if len(candidate_folders) == 1:
        label_folder = candidate_folders[0]
    else:
        try:
            label_room = room_ids[row['loc']]
        except KeyError:
            print(f"{participant_id} - Skipping row {index} because no location provided: {row['loc']} with {len(candidate_folders)} filenames")
            continue
        try:
            label_folder = [bf for bf in candidate_folders if label_room in bf][0]
        except IndexError:
            print(f"{participant_id} - Skipping row {index} because no file found for location {row['loc']}")
            continue

    # Convert the label folder into folder for my path
    label_parts = Path(label_folder).parts
    sub_part_i = [bfi for bfi, bf in enumerate(label_parts) if bf.startswith('sub-')][0]
    video_path = data_folder / Path(*label_parts[sub_part_i:])

    # Read the FPS of the original video
    cap = cv2.VideoCapture(str(rdsf_path / Path(*Path(label_folder).parts[1:])))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    video_timestamp = int(label_parts[-1].split('uk_data_')[1].split('_rgb_video')[0])

    # Find the csv files from the video folder
    csv_files = [bf for bf in video_path.glob('*.npz')]
    # Find the CSV with the timestamps that matches the label
    matching_data = []
    matching_frame = []
    matching_time = []
    if debug_sts:
        plt.figure('Combined', figsize=[15, 4])
        plt.clf()
        gs = None#gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2.5])
        plt.figure('Matching tracklets')
        plt.clf()
    else:
        gs = None

    for ci, csv_file in enumerate(csv_files):
        # data = np.genfromtxt(csv_file, delimiter=',')
        data = np.load(csv_file)['data']
        if len(data.shape) < 2:
            continue

        # Convert frames to timestamp.
        vid_frames = data[:, -1]
        vid_time = vid_frames/fps + video_timestamp/1000

        # Find the sts frame
        sts_frame = np.searchsorted(vid_time, sts_timestamp)
        # Make sure the frame falls within the interval
        if sts_frame > len(vid_frames)-1 or sts_frame==0:
            continue
        # Make sure that the frames are consecutive
        diff = np.diff(vid_time)[sts_frame-1]
        if diff <= 0.5:#1/fps*1.1:
            matching_data.append(data)
            matching_frame.append(sts_frame)
            matching_time.append(vid_time)

        if debug_sts:
            plt.plot(vid_time, np.zeros_like(vid_time)+ci, '.-', label=f"{csv_file.name} - {diff:.2f}")

    if debug_sts:
        plt.axvline(x=sts_timestamp)
        plt.legend()
        # plt.savefig(f'{index}_tracklets.png')


    # There can only be one or two people in front of the camera
    if len(matching_data) not in [1, 2]:
        print(f'{participant_id} - #{index} - Found {len(matching_data)} matching tracklets. Not implemented yet')
        continue
    # Pick the correct tracklet based on the location label
    elif len(matching_data) == 1:
        id_target = 0
    elif len(matching_data) == 2:
        frame_loc = row['frame_loc']
        sts_slice0 = slice(matching_frame[0], round(matching_frame[0] + row['sts_final_attempt_duration duration'] * fps))
        sts_slice1 = slice(matching_frame[1], round(matching_frame[1] + row['sts_final_attempt_duration duration'] * fps))
        x0 = matching_data[0][sts_slice0, 0]
        x0 = x0[x0 != 0].mean()
        y0 = matching_data[0][sts_slice0, 1]
        y0 = y0[y0 != 0].mean()
        x1 = matching_data[1][sts_slice1, 0]
        x1 = x1[x1 != 0].mean()
        y1 = matching_data[1][sts_slice1, 1]
        y1 = y1[y1 != 0].mean()
        if frame_loc == 'left':
            id_target = 0 if x0 < x1 else 1
        elif frame_loc == 'right':
            id_target = 0 if x0 > x1 else 1
        elif frame_loc == 'above':
            id_target = 0 if y0 < y1 else 1
        elif frame_loc == 'below':
            id_target = 0 if y0 > y1 else 1
        else:
            print(f'{participant_id} - #{index} Frame location not set, skipping this multiple entry')
            continue

        if debug_sts:
            plt.figure('Frame location')
            plt.clf()
            plt.plot(x0,y0,'ok',label='0')
            plt.plot(x1,y1,'xr',label='1')
            plt.title(f'frame_loc: {frame_loc}, id_target: {id_target}')
            plt.legend()
            # plt.savefig(f'{index}_locations.png')

    # Process the bounding box data
    data = matching_data[id_target]
    frame = matching_frame[id_target]
    extra_time_bf = int(2*fps)  # Seconds
    extra_time_af = int((max_sts_duration+2)*fps)  # Seconds
    # Remove zeros
    sts_slice = slice((frame - extra_time_bf), round(frame + extra_time_af))

    # hips: 9 8 12
    # head: 0 15 16 17 18
    # shoulders: 2 1 5
    # head_joints = [0, 15, 16, 17, 18, 2, 1, 5]
    # head_joints = [0, 15, 16, 17, 18, 2, 1, 5]
    if zz_which_joints == 0:
        head_joints = [0, 15, 16, 17, 18, 2, 1, 5, 9, 8, 12]
    elif zz_which_joints == 1:
        head_joints = [0, 15, 16, 17, 18, 2, 1, 5]
    elif zz_which_joints == 2:
        head_joints = [0, 15, 16, 17, 18]
    elif zz_which_joints == 3:
        head_joints = [0]
    elif zz_which_joints == 4:
        head_joints = [2, 1, 5]
    elif zz_which_joints == 5:
        head_joints = [9, 8, 12]
    # head_joints = [0, 15, 16, 17, 18]
    all_masks = []
    all_joints = []
    stop = False
    non_valid = []
    for head_joint in head_joints:
        y_joint = data[sts_slice, head_joint * 2 + 1]
        mask = y_joint == 0
        non_valid.append(np.sum(mask))

    best_head = np.argmin(non_valid)
    head_joint = head_joints[best_head]

    y_joint = data[sts_slice, head_joint * 2 + 1]
    mask = y_joint == 0
    if mask.sum() / len(mask) > min_mask or len(mask) < 11:  # 0.4:
        # print(f'#{index} - Invalid/valid poses is {mask.sum() / len(mask) * 100:.1f}%. Nothing to do here')
        continue

    # Replace missing values
    y_joint[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_joint[~mask])

    # mask = np.any(np.vstack(all_masks), axis=0)
    # if mask.sum()/len(mask) > 0.1: #0.4:
    #     print(f'#{index} - Invalid/valid poses is {mask.sum()/len(mask)*100:.1f}%. Nothing to do here')
    #     continue

    y_head = y_joint.copy()

    # y_head = data[sts_slice, 1]
    # # y_head = (data[sts_slice, 9*2+1] + data[sts_slice, 8*2+1] + data[sts_slice, 12*2+1])/3
    # y_stern = data[sts_slice, 3]
    # mask = np.any(np.vstack((y_head==0, y_stern==0)), axis=0)#y_head==0
    # if mask.sum()/len(mask) > 0.1: #0.4:
    #     print(f'#{index} - Invalid/valid poses is {mask.sum()/len(mask)*100:.1f}%. Nothing to do here')
    #     continue
    #
    # if len(y_head) < 11:
    #     print(f'#{index} - Skipping because not enough data points: {len(y_head)}')
    #     continue
    # y_head[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_head[~mask])
    # y_stern[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_stern[~mask])
    # # np.where(y_head == 0)
    # # plt.figure()
    # # plt.plot(y_head_sts)
    #
    # # Calculate the calibration
    # mean_neck_height = np.mean(np.abs(y_stern - y_head))
    # # mean_neck_height2 = np.median(np.abs(y_stern - y_head))

    # Calculate the STS speed
    time = matching_time[id_target][sts_slice]
    if np.any(np.diff(time) > 1/fps*1.1):
        print(f'{participant_id} - #{index} - Skipping because of non-contiguous clip: {np.diff(time).max():.1f} sec')
        continue
    cy = -y_head  # Use the y coordinate of the upper edge of the 3D bounding box
    # cy = medfilt(cy, kernel_size=3)
    cy = medfilt(cy, kernel_size=medfilt_k)
    direction = 'stand up' if row['act_or_sed'] == 'sit_to_stand' else 'sit down'
    # speed, duration, snr, extent, time_i = speed_of_ascent(cy, time, direction, debug=debug_sts, smooth_kernel=25, smooth_poly=3)
    # speed, duration, snr, extent, time_i, time_end = speed_of_ascent(cy, time, direction, debug=debug_sts,
    #                                                        smooth_kernel=smooth_kernel, smooth_poly=smooth_poly, gs=gs, zct=zero_cross_thr,
    #                                                        label=[matching_time[id_target][frame], row['sts_final_attempt_duration duration']])
    # # w, h = skeleton_sizes(data[sts_slice, :][time_end, :])
    # # if time_end > 0:
    # #     h = np.mean(np.array(list(map(skeleton_sizes, data[sts_slice, :][time_i:time_end]))), axis=0)[1]
    # # else:
    # w, h = skeleton_sizes(data[sts_slice, :][time_end, :])
    # df.at[index, 'sts_speed'] = speed/h*1.7 #m/s
    # df.at[index, 'subj_height'] = h
    # df.at[index, 'sts_duration'] = duration
    # df.at[index, 'snr'] = snr
    # df.at[index, 'extent'] = extent
    # df.at[index, 'neck_height'] = np.abs(y_stern[time_i] - y_head[time_i])
    # df.at[index, 'neck_height2'] = mean_neck_height




    ## REDUCE SKELETON QUALITY
    data_coarse = data.copy()
    data_coarse[:, :-1] = (data_coarse[:, :-1] // 2) * 2

    pd_or_c = row['PD_or_C']
    excel_id = row['Transition ID']
    new_id = row['New ID number']

    # file_name = f'standup_{uid}_PID_{new_id}_{pd_or_c}_EXID_{excel_id}.csv'
    file_name = f'Pt{new_id}_{pd_or_c}_n_{excel_id:04d}.csv'
    fmt = '%g'
    header = 'frame number, x0, y0, x1, y1, ..., x24, y24 (25 default joints as per OpenPose doc: https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html )'
    frame_number = data[sts_slice, -1]
    if frame_number.size == 0:
        continue
    frame_number = frame_number - frame_number[0]
    bf1 = np.hstack((frame_number[:, None], data[sts_slice, :-1]))
    bf2 = np.hstack((frame_number[:, None], data_coarse[sts_slice, :-1]))
    np.savetxt(output_folder / 'fine' / file_name, bf1, fmt, delimiter=',', header=header)
    np.savetxt(output_folder / 'coarse' / file_name, bf2, fmt, delimiter=',', header=header)
    uid += 1



    if debug_sts:
        plt.figure('Combined')
        # plt.gcf().set_size_inches([15, 7])
        # plt.subplot(gs[2])
        # # plt.subplot(3, 1, 3)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Pixels/s')
        # plt.grid(visible=True, which='major', linestyle=':')
        # plt.tight_layout()
        # plt.savefig(f'{index}_derivative.png')

        # plt.figure('Bounding box')
        # plt.gcf().set_size_inches([15, 7])
        # plt.subplot(gs[1])
        # plt.xlabel('Time (s)')
        # plt.ylabel('Pixels')
        # plt.grid(visible=True, which='major', linestyle=':')
        # plt.tight_layout()
        # xlim = plt.gca().get_xlim()
        # plt.savefig(f'{index}_bbox.png')

        # # Make figure of skeletons
        # Plot the head trajectory
        stretch = 80
        n_skel = 15
        time_skel = (time - time[0])*stretch

        # plt.figure('Skeleton', figsize=[15, 7])
        # plt.clf()
        # plt.subplot(gs[0])
        # plt.plot(time_skel, -cy, color=[0.7, 0.7, 0.7], zorder=10, linewidth=2, linestyle='--', label='Head trajectory')

        skeleton_data = data[sts_slice, :]
        sub_i = np.linspace(0, len(skeleton_data)-1, n_skel).astype(int)
        skeleton_data = skeleton_data[sub_i]
        sub_time = time_skel[sub_i]
        cmap = matplotlib.cm.get_cmap('gist_ncar')
        for fi in range(n_skel):
            this_skel = skeleton_data[fi, ].copy()
            bf = this_skel[:-1].reshape((-1, 2))
            mk = np.any(bf == 0, axis=1)[:, None]
            bf[np.hstack((mk, mk))] = np.nan
            this_skel[0:-1:2] = this_skel[0:-1:2] - np.nanmean(bf[:, 0]) + sub_time[fi]
            plot_skeleton(this_skel, color=cmap(fi/n_skel), marker='o', markersize=3)

        plt.gca().invert_yaxis()
        plt.axis('equal')
        # plt.xlim(-3, 3)
        # plt.ylim(-cy.min(), -cy.max())
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([], [])
        plt.yticks([], [])

        plt.legend()
        plt.tight_layout()

        plt.savefig(output_folder / 'img' / f'{file_name}.png', dpi=200)
        # sdf


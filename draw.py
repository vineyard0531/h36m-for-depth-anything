import pickle
import math
import matplotlib.pyplot as plt

def distance_between_points(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

# 从 pkl 文件加载 joints_3d 数据
with open('/home/tanli/Code/DepthAnything/h36m_joint_3d_pkl/s_01_act_02_subact_02_ca_01_joints_3d.pkl', 'rb') as f:
    joints_3d_dict = pickle.load(f)

#with open('/home/tanli/Code/DepthAnything/Depth-Anything/metric_depth/z_smooth_cropped_v1_pkl/z_s_01_act_02_subact_02_ca_01_smooth_cropped_v1.pkl', 'rb') as f:
#    z_v1_dict = pickle.load(f)

with open('/home/tanli/Code/DepthAnything/Depth-Anything-V2-main/Depth-Anything-V2-main/metric_depth/z_s_01_act_02_subact_02_ca_01_smooth_cropped_v2.pkl', 'rb') as f:
    z_v2_dict = pickle.load(f)

# 存储每一帧的距离数据
frames = []
distances_gt = {
    '0_16': [],
    '17_25': [],
    '1_6': [],
    '0_15': [],
    '1_2': [],
    '2_3': [],
    '6_7': [],
    '7_8': [],
    '17_18': [],
    '18_19': [],
    '25_26': [],
    '26_27': [],
    'leg_l': [],
    'leg_r': [],
    'arm_l': [],
    'arm_r': [],
    'h_1': [],
    'h_2': []
}

distances_dp_v1 = {
    '0_16': [],
    '17_25': [],
    '1_6': [],
    '0_15': [],
    '1_2': [],
    '2_3': [],
    '6_7': [],
    '7_8': [],
    '17_18': [],
    '18_19': [],
    '25_26': [],
    '26_27': [],
    'leg_l': [],
    'leg_r': [],
    'arm_l': [],
    'arm_r': [],
    'h_1': [],
    'h_2': []
}

distances_dp_v2 = {
    '0_16': [],
    '17_25': [],
    '1_6': [],
    '0_15': [],
    '1_2': [],
    '2_3': [],
    '6_7': [],
    '7_8': [],
    '17_18': [],
    '18_19': [],
    '25_26': [],
    '26_27': [],
    'leg_l': [],
    'leg_r': [],
    'arm_l': [],
    'arm_r': [],
    'h_1': [],
    'h_2': []
}
for i in range(1, 1613):
    image_name1 = f's_01_act_02_subact_02_ca_01/s_01_act_02_subact_02_ca_01_{i:06d}.jpg'
    image_name2 = f's_01_act_02_subact_02_ca_01_{i:06d}.jpg'

#    if image_name1 in joints_3d_dict and image_name2 in z_v1_dict and image_name2 in z_v2_dict:
    joints_3d = joints_3d_dict[image_name1]
    #z_v1 = z_v1_dict[image_name2]
    z_v2 = z_v2_dict[image_name2]

    # 将 joints_3d 中的每个元素乘以 0.001 并保留五位小数
    if joints_3d is not None:
        joints_3d = [[round(coordinate * 0.001, 5) for coordinate in point] for point in joints_3d]
    '''
    #v1
    depth_points_0_v1 = (joints_3d[0][0], joints_3d[0][1], z_v1[0]) 
    depth_points_1_v1 = (joints_3d[1][0], joints_3d[1][1], z_v1[1])
    depth_points_2_v1 = (joints_3d[2][0], joints_3d[2][1], z_v1[2])
    depth_points_3_v1 = (joints_3d[3][0], joints_3d[3][1], z_v1[3])
    depth_points_6_v1 = (joints_3d[4][0], joints_3d[4][1], z_v1[4])
    depth_points_7_v1 = (joints_3d[5][0], joints_3d[5][1], z_v1[5])
    depth_points_8_v1 = (joints_3d[6][0], joints_3d[6][1], z_v1[6])
    depth_points_16_v1 =(joints_3d[8][0], joints_3d[8][1], z_v1[8])
    depth_points_15_v1 =(joints_3d[10][0], joints_3d[10][1], z_v1[10])
    depth_points_17_v1 =(joints_3d[11][0], joints_3d[11][1], z_v1[11])
    depth_points_18_v1 =(joints_3d[12][0], joints_3d[12][1], z_v1[12])
    depth_points_19_v1 =(joints_3d[13][0], joints_3d[13][1], z_v1[13])
    depth_points_25_v1 =(joints_3d[14][0], joints_3d[14][1], z_v1[14])
    depth_points_26_v1 =(joints_3d[15][0], joints_3d[15][1], z_v1[15])
    depth_points_27_v1 =(joints_3d[16][0], joints_3d[16][1], z_v1[16])

    '''
    #v2
    depth_points_0_v2 = (joints_3d[0][0], joints_3d[0][1], z_v2[0]) 
    depth_points_1_v2 = (joints_3d[1][0], joints_3d[1][1], z_v2[1])
    depth_points_2_v2 = (joints_3d[2][0], joints_3d[2][1], z_v2[2])
    depth_points_3_v2 = (joints_3d[3][0], joints_3d[3][1], z_v2[3])
    depth_points_6_v2 = (joints_3d[4][0], joints_3d[4][1], z_v2[4])
    depth_points_7_v2 = (joints_3d[5][0], joints_3d[5][1], z_v2[5])
    depth_points_8_v2 = (joints_3d[6][0], joints_3d[6][1], z_v2[6])
    depth_points_16_v2 =(joints_3d[8][0], joints_3d[8][1], z_v2[8])
    depth_points_15_v2 =(joints_3d[10][0], joints_3d[10][1], z_v2[10])
    depth_points_17_v2 =(joints_3d[11][0], joints_3d[11][1], z_v2[11])
    depth_points_18_v2 =(joints_3d[12][0], joints_3d[12][1], z_v2[12])
    depth_points_19_v2 =(joints_3d[13][0], joints_3d[13][1], z_v2[13])
    depth_points_25_v2 =(joints_3d[14][0], joints_3d[14][1], z_v2[14])
    depth_points_26_v2 =(joints_3d[15][0], joints_3d[15][1], z_v2[15])
    depth_points_27_v2 =(joints_3d[16][0], joints_3d[16][1], z_v2[16])
    #'''
    
   
    #h36m
    gt_points_0 = joints_3d[0] 
    gt_points_1 = joints_3d[1]
    gt_points_2 = joints_3d[2]
    gt_points_3 = joints_3d[3]
    gt_points_6 = joints_3d[4]
    gt_points_7 = joints_3d[5]
    gt_points_8 = joints_3d[6]
    gt_points_16 = joints_3d[8]
    gt_points_15 = joints_3d[10]
    gt_points_17 = joints_3d[11]
    gt_points_18 = joints_3d[12]
    gt_points_19 = joints_3d[13]
    gt_points_25 = joints_3d[14]
    gt_points_26 = joints_3d[15]
    gt_points_27 = joints_3d[16]
    # h36m distances
    distance_gt = {
        '0_16': round(distance_between_points(gt_points_0, gt_points_16), 5),
        '17_25': round(distance_between_points(gt_points_17, gt_points_25), 5),
        '1_6': round(distance_between_points(gt_points_1, gt_points_6), 5),
        '0_15': round(distance_between_points(gt_points_0, gt_points_15), 5),
        '1_2': round(distance_between_points(gt_points_1, gt_points_2), 5),
        '2_3': round(distance_between_points(gt_points_2, gt_points_3), 5),
        '6_7': round(distance_between_points(gt_points_6, gt_points_7), 5),
        '7_8': round(distance_between_points(gt_points_7, gt_points_8), 5),
        '17_18': round(distance_between_points(gt_points_17, gt_points_18), 5),
        '18_19': round(distance_between_points(gt_points_18, gt_points_19), 5),
        '25_26': round(distance_between_points(gt_points_25, gt_points_26), 5),
        '26_27': round(distance_between_points(gt_points_26, gt_points_27), 5)
    }

    distance_gt['leg_l'] = round(distance_gt['1_2'] + distance_gt['2_3'], 5)
    distance_gt['leg_r'] = round(distance_gt['6_7'] + distance_gt['7_8'], 5)
    distance_gt['arm_l'] = round(distance_gt['17_18'] + distance_gt['18_19'], 5)
    distance_gt['arm_r'] = round(distance_gt['25_26'] + distance_gt['26_27'], 5)
    distance_gt['h_1'] = round(distance_gt['0_15'] + distance_gt['leg_l'], 5)
    distance_gt['h_2'] = round(distance_gt['0_15'] + distance_gt['leg_r'], 5)
    '''
    # depth anything distances v1
    distance_dp_v1 = {
        '0_16': round(distance_between_points(depth_points_0_v1, depth_points_16_v1), 5),
        '17_25': round(distance_between_points(depth_points_17_v1, depth_points_25_v1), 5),
        '1_6': round(distance_between_points(depth_points_1_v1, depth_points_6_v1), 5),
        '0_15': round(distance_between_points(depth_points_0_v1, depth_points_15_v1), 5),
        '1_2': round(distance_between_points(depth_points_1_v1, depth_points_2_v1), 5),
        '2_3': round(distance_between_points(depth_points_2_v1, depth_points_3_v1), 5),
        '6_7': round(distance_between_points(depth_points_6_v1, depth_points_7_v1), 5),
        '7_8': round(distance_between_points(depth_points_7_v1, depth_points_8_v1), 5),
        '17_18': round(distance_between_points(depth_points_17_v1, depth_points_18_v1), 5),
        '18_19': round(distance_between_points(depth_points_18_v1, depth_points_19_v1), 5),
        '25_26': round(distance_between_points(depth_points_25_v1, depth_points_26_v1), 5),
        '26_27': round(distance_between_points(depth_points_26_v1, depth_points_27_v1), 5)
    }

    distance_dp_v1['leg_l'] = round(distance_dp_v1['1_2'] + distance_dp_v1['2_3'], 5)
    distance_dp_v1['leg_r'] = round(distance_dp_v1['6_7'] + distance_dp_v1['7_8'], 5)
    distance_dp_v1['arm_l'] = round(distance_dp_v1['17_18'] + distance_dp_v1['18_19'], 5)
    distance_dp_v1['arm_r'] = round(distance_dp_v1['25_26'] + distance_dp_v1['26_27'], 5)
    distance_dp_v1['h_1'] = round(distance_dp_v1['0_15'] + distance_dp_v1['leg_l'], 5)
    distance_dp_v1['h_2'] = round(distance_dp_v1['0_15'] + distance_dp_v1['leg_r'], 5)
    '''
    # depth anything distances v2
    distance_dp_v2 = {
        '0_16': round(distance_between_points(depth_points_0_v2, depth_points_16_v2), 5),
        '17_25': round(distance_between_points(depth_points_17_v2, depth_points_25_v2), 5),
        '1_6': round(distance_between_points(depth_points_1_v2, depth_points_6_v2), 5),
        '0_15': round(distance_between_points(depth_points_0_v2, depth_points_15_v2), 5),
        '1_2': round(distance_between_points(depth_points_1_v2, depth_points_2_v2), 5),
        '2_3': round(distance_between_points(depth_points_2_v2, depth_points_3_v2), 5),
        '6_7': round(distance_between_points(depth_points_6_v2, depth_points_7_v2), 5),
        '7_8': round(distance_between_points(depth_points_7_v2, depth_points_8_v2), 5),
        '17_18': round(distance_between_points(depth_points_17_v2, depth_points_18_v2), 5),
        '18_19': round(distance_between_points(depth_points_18_v2, depth_points_19_v2), 5),
        '25_26': round(distance_between_points(depth_points_25_v2, depth_points_26_v2), 5),
        '26_27': round(distance_between_points(depth_points_26_v2, depth_points_27_v2), 5)
    }

    distance_dp_v2['leg_l'] = round(distance_dp_v2['1_2'] + distance_dp_v2['2_3'], 5)
    distance_dp_v2['leg_r'] = round(distance_dp_v2['6_7'] + distance_dp_v2['7_8'], 5)
    distance_dp_v2['arm_l'] = round(distance_dp_v2['17_18'] + distance_dp_v2['18_19'], 5)
    distance_dp_v2['arm_r'] = round(distance_dp_v2['25_26'] + distance_dp_v2['26_27'], 5)
    distance_dp_v2['h_1'] = round(distance_dp_v2['0_15'] + distance_dp_v2['leg_l'], 5)
    distance_dp_v2['h_2'] = round(distance_dp_v2['0_15'] + distance_dp_v2['leg_r'], 5)
    #'''
   
    # 记录每一帧的结果
    frames.append(i)
    for key in distances_gt:
        distances_gt[key].append(distance_gt[key])
        #distances_dp_v1[key].append(distance_dp_v1[key])
        distances_dp_v2[key].append(distance_dp_v2[key])

# 创建图像和子图
fig, axs = plt.subplots(4, 2, figsize=(15, 20))

# 设置每个子图的标题和绘制内容
titles = [
    'h_1 (Head to Foot, Left)', 'h_2 (Head to Foot, Right)',
    '17_25 (jianbu)', '1_6 (kuabu)',
    'leg_l (Left Leg)', 'leg_r (Right Leg)',
    'arm_l (Left Arm, Lower)', 'arm_r (Right Arm, Lower)'
]

# 绘制每个子图
keys = ['h_1', 'h_2', '17_25', '1_6', 'leg_l', 'leg_r', 'arm_l', 'arm_r']
for i, ax in enumerate(axs.flat):
    key = keys[i]
    #ax.plot(frames, distances_dp_v1[key], label='Depth Anything V1(dp)', color='red')
    ax.plot(frames, distances_dp_v2[key], label='Depth Anything V2(dp)', color='green')
    ax.plot(frames, distances_gt[key], label='Ground Truth (gt)', color='blue')
    ax.set_title(titles[i])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.legend()

# 调整布局并显示图像

plt.tight_layout()
plt.savefig('pts_s_01_act_02_subact_02_ca_01_s_v2.jpg') 
plt.show()

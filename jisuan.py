import pickle
import math

def distance_between_points(p1, p2):
    # p1 和 p2 是两个三维点，格式为 (x, y, z)
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

# 从 pkl 文件加载 joints_3d 数据
with open('/home/tanli/Code/DepthAnything/h36m_joint_3d_pkl/s_01_act_02_subact_02_ca_01_joints_3d.pkl', 'rb') as f:
    joints_3d_dict = pickle.load(f)

with open('/home/tanli/Code/DepthAnything/Depth-Anything/metric_depth/z_smooth_cropped_v1_pkl/z_s_01_act_02_subact_02_ca_01_smooth_cropped_v1.pkl', 'rb') as f:
    z_v1_dict = pickle.load(f)

with open('/home/tanli/Code/DepthAnything/Depth-Anything-V2-main/Depth-Anything-V2-main/metric_depth/z_s_01_act_02_subact_02_ca_01_smooth_cropped_v2.pkl', 'rb') as f:
    z_v2_dict = pickle.load(f)

#'s_01_act_02_subact_02_ca_02/s_01_act_02_subact_02_ca_02_000001.jpg'
for i in range(1, 1612):
    image_name1 = f's_01_act_02_subact_02_ca_01/s_01_act_02_subact_02_ca_01_{i:06d}.jpg'
    image_name2 = f's_01_act_02_subact_02_ca_01_{i:06d}.jpg'
    print(f's_01_act_02_subact_02_ca_01_{i:06d}.jpg:\n')

    # 获取该图片的 joints_3d 和 z 数据
    #if image_name1 in joints_3d_dict and image_name2 in z_v1_dict and image_name2 in z_v2_dict:
    joints_3d = joints_3d_dict[image_name1]
    z_v1 = z_v1_dict[image_name2]
    z_v2 = z_v2_dict[image_name2]

    # 将 joints_3d 中的每个元素乘以 0.001 并保留五位小数
    if joints_3d is not None:
    # 确保 joints_3d 是二维数组，并且对每个值进行缩放和保留5位小数
        joints_3d = [[round(coordinate * 0.001, 5) for coordinate in point] for point in joints_3d]

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

    #'''
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

    # h36m
    distance_gt_0_16 = round(distance_between_points(gt_points_0, gt_points_16), 5)
    distance_gt_17_25 = round(distance_between_points(gt_points_17, gt_points_25), 5)
    distance_gt_1_6 = round(distance_between_points(gt_points_1, gt_points_6), 5)
    distance_gt_0_15 = round(distance_between_points(gt_points_0, gt_points_15), 5)
    distance_gt_1_2 = round(distance_between_points(gt_points_1, gt_points_2), 5)
    distance_gt_2_3 = round(distance_between_points(gt_points_2, gt_points_3), 5)
    distance_gt_6_7 = round(distance_between_points(gt_points_6, gt_points_7), 5)
    distance_gt_7_8 = round(distance_between_points(gt_points_7, gt_points_8), 5)
    distance_gt_17_18 = round(distance_between_points(gt_points_17, gt_points_18), 5)
    distance_gt_18_19 = round(distance_between_points(gt_points_18, gt_points_19), 5)
    distance_gt_25_26 = round(distance_between_points(gt_points_25, gt_points_26), 5)
    distance_gt_26_27 = round(distance_between_points(gt_points_26, gt_points_27), 5)

    distance_gt_leg_l = round(distance_gt_1_2 + distance_gt_2_3, 5)
    distance_gt_leg_r = round(distance_gt_6_7 + distance_gt_7_8, 5)

    distance_gt_arm_l = round(distance_gt_17_18 + distance_gt_18_19, 5)
    distance_gt_arm_r = round(distance_gt_25_26 + distance_gt_26_27, 5)

    distance_gt_h_1 = round(distance_gt_0_15 + distance_gt_leg_l, 5)
    distance_gt_h_2 = round(distance_gt_0_15 + distance_gt_leg_r, 5)

    # depth anything v1
    distance_dp_0_16_v1 = round(distance_between_points(depth_points_0_v1, depth_points_16_v1), 5)
    distance_dp_17_25_v1 = round(distance_between_points(depth_points_17_v1, depth_points_25_v1), 5)
    distance_dp_1_6_v1 = round(distance_between_points(depth_points_1_v1, depth_points_6_v1), 5)
    distance_dp_0_15_v1 = round(distance_between_points(depth_points_0_v1, depth_points_15_v1), 5)
    distance_dp_1_2_v1 = round(distance_between_points(depth_points_1_v1, depth_points_2_v1), 5)
    distance_dp_2_3_v1 = round(distance_between_points(depth_points_2_v1, depth_points_3_v1), 5)
    distance_dp_6_7_v1 = round(distance_between_points(depth_points_6_v1, depth_points_7_v1), 5)
    distance_dp_7_8_v1 = round(distance_between_points(depth_points_7_v1, depth_points_8_v1), 5)
    distance_dp_17_18_v1 = round(distance_between_points(depth_points_17_v1, depth_points_18_v1), 5)
    distance_dp_18_19_v1 = round(distance_between_points(depth_points_18_v1, depth_points_19_v1), 5)
    distance_dp_25_26_v1 = round(distance_between_points(depth_points_25_v1, depth_points_26_v1), 5)
    distance_dp_26_27_v1 = round(distance_between_points(depth_points_26_v1, depth_points_27_v1), 5)

    distance_dp_leg_l_v1 = round(distance_dp_1_2_v1 + distance_dp_2_3_v1, 5)
    distance_dp_leg_r_v1 = round(distance_dp_6_7_v1 + distance_dp_7_8_v1, 5)

    distance_dp_arm_l_v1 = round(distance_dp_17_18_v1 + distance_dp_18_19_v1, 5)
    distance_dp_arm_r_v1 = round(distance_dp_25_26_v1 + distance_dp_26_27_v1, 5)

    distance_dp_h_1_v1 = round(distance_dp_0_15_v1 + distance_dp_leg_l_v1, 5)
    distance_dp_h_2_v1 = round(distance_dp_0_15_v1 + distance_dp_leg_r_v1, 5)
    #'''
    # depth anything v2
    distance_dp_0_16_v2 = round(distance_between_points(depth_points_0_v2, depth_points_16_v2), 5)
    distance_dp_17_25_v2 = round(distance_between_points(depth_points_17_v2, depth_points_25_v2), 5)
    distance_dp_1_6_v2 = round(distance_between_points(depth_points_1_v2, depth_points_6_v2), 5)
    distance_dp_0_15_v2 = round(distance_between_points(depth_points_0_v2, depth_points_15_v2), 5)
    distance_dp_1_2_v2 = round(distance_between_points(depth_points_1_v2, depth_points_2_v2), 5)
    distance_dp_2_3_v2 = round(distance_between_points(depth_points_2_v2, depth_points_3_v2), 5)
    distance_dp_6_7_v2 = round(distance_between_points(depth_points_6_v2, depth_points_7_v2), 5)
    distance_dp_7_8_v2 = round(distance_between_points(depth_points_7_v2, depth_points_8_v2), 5)
    distance_dp_17_18_v2 = round(distance_between_points(depth_points_17_v2, depth_points_18_v2), 5)
    distance_dp_18_19_v2 = round(distance_between_points(depth_points_18_v2, depth_points_19_v2), 5)
    distance_dp_25_26_v2 = round(distance_between_points(depth_points_25_v2, depth_points_26_v2), 5)
    distance_dp_26_27_v2 = round(distance_between_points(depth_points_26_v2, depth_points_27_v2), 5)

    distance_dp_leg_l_v2 = round(distance_dp_1_2_v2 + distance_dp_2_3_v2, 5)
    distance_dp_leg_r_v2 = round(distance_dp_6_7_v2 + distance_dp_7_8_v2, 5)

    distance_dp_arm_l_v2 = round(distance_dp_17_18_v2 + distance_dp_18_19_v2, 5)
    distance_dp_arm_r_v2 = round(distance_dp_25_26_v2 + distance_dp_26_27_v2, 5)

    distance_dp_h_1_v2 = round(distance_dp_0_15_v2 + distance_dp_leg_l_v2, 5)
    distance_dp_h_2_v2 = round(distance_dp_0_15_v2 + distance_dp_leg_r_v2, 5)
    #'''



    #差值
    d_h1_v1 = round(distance_gt_h_1 - distance_dp_h_1_v1, 5)
    d_h2_v1 = round(distance_gt_h_2 - distance_dp_h_2_v1, 5)
    d_17_25_v1 = round(distance_gt_17_25 - distance_dp_17_25_v1, 5)
    d_0_13_v1 = round(distance_gt_0_16 - distance_dp_0_16_v1, 5)
    d_1_6_v1 = round(distance_gt_1_6 - distance_dp_1_6_v1, 5)
    d_0_15_v1 = round(distance_gt_0_15 - distance_dp_0_15_v1, 5)
    d_leg_l_v1 = round(distance_gt_leg_l - distance_dp_leg_l_v1, 5)
    d_leg_r_v1 = round(distance_gt_leg_r - distance_dp_leg_r_v1, 5)
    d_arm_l_v1 = round(distance_gt_arm_l - distance_dp_arm_l_v1, 5)
    d_arm_r_v1 = round(distance_gt_arm_r - distance_dp_arm_r_v1, 5)
    #'''
    d_h1_v2 = round(distance_gt_h_1 - distance_dp_h_1_v2, 5)
    d_h2_v2 = round(distance_gt_h_2 - distance_dp_h_2_v2, 5)
    d_17_25_v2 = round(distance_gt_17_25 - distance_dp_17_25_v2, 5)
    d_0_13_v2 = round(distance_gt_0_16 - distance_dp_0_16_v2, 5)
    d_1_6_v2 = round(distance_gt_1_6 - distance_dp_1_6_v2, 5)
    d_0_15_v2 = round(distance_gt_0_15 - distance_dp_0_15_v2, 5)
    d_leg_l_v2 = round(distance_gt_leg_l - distance_dp_leg_l_v2, 5)
    d_leg_r_v2 = round(distance_gt_leg_r - distance_dp_leg_r_v2, 5)
    d_arm_l_v2 = round(distance_gt_arm_l - distance_dp_arm_l_v2, 5)
    d_arm_r_v2 = round(distance_gt_arm_r - distance_dp_arm_r_v2, 5)
    #'''



    
    print('所有差值为gt-depthanything的值\n')

    print(f"d_h1_v1: {d_h1_v1}  d_h2_v1: {d_h2_v1}  d_17_25_v1: {d_17_25_v1}  d_0_13_v1: {d_0_13_v1}  d_1_6_v1: {d_1_6_v1}  d_0_15_v1: {d_0_15_v1}  d_leg_l_v1: {d_leg_l_v1}  d_leg_r_v1: {d_leg_r_v1}  d_arm_l_v1: {d_arm_l_v1}  d_arm_r_v1: {d_arm_r_v1}")
    print(f"d_h1_v2: {d_h1_v2}  d_h2_v2: {d_h2_v2}  d_17_25_v2: {d_17_25_v2}  d_0_13_v2: {d_0_13_v2}  d_1_6_v2: {d_1_6_v2}  d_0_15_v2: {d_0_15_v2}  d_leg_l_v2: {d_leg_l_v2}  d_leg_r_v2: {d_leg_r_v2}  d_arm_l_v2: {d_arm_l_v2}  d_arm_r_v2: {d_arm_r_v2}")
    
    print('\n')
    print(f"dp_v1身高1(左腿)是: {distance_dp_h_1_v1}  dp_v1身高2(右腿)是：{distance_dp_h_2_v1}  dp_v1肩部：{distance_dp_17_25_v1}  dp_v1胯部：{distance_dp_1_6_v1}  dp_v1左腿:{distance_dp_leg_l_v1}  dp_v1右腿:{distance_dp_leg_r_v1}  dp_v1左臂:{distance_dp_arm_l_v1}  dp_v1右臂:{distance_dp_arm_r_v1}")
    print(f"dp_v2身高1(左腿)是: {distance_dp_h_1_v2}  dp_v2身高2(右腿)是：{distance_dp_h_2_v2}  dp_v2肩部：{distance_dp_17_25_v2}  dp_v2胯部：{distance_dp_1_6_v2}  dp_v2左腿:{distance_dp_leg_l_v2}  dp_v2右腿:{distance_dp_leg_r_v2}  dp_v2左臂:{distance_dp_arm_l_v2}  dp_v2右臂:{distance_dp_arm_r_v2}")
    
    print(f"gt身高1(左腿)是:    {distance_gt_h_1}  gt身高2(右腿)是：   {distance_gt_h_2}  gt肩部：   {distance_gt_17_25}  gt胯部：   {distance_gt_1_6}  gt左腿:{distance_gt_leg_l}  gt右腿:   {distance_gt_leg_r}  gt左臂:{distance_gt_arm_l}  gt右臂:   {distance_gt_arm_r}")
    print('\n')
    










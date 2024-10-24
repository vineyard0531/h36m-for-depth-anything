import os
import cv2
import numpy as np
import pickle

dst_height, dst_width = 392, 518

# 读取并筛选符合条件的 joints_2d 的函数
def extract_joints_2d_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 筛选出符合条件的数据
    filtered_data = [d for d in data if d['action'] == 2 and d['subaction'] == 2 and d['camera_id'] == 2 and d['subject'] == 5]
    
    # 提取所有 joints_2d 数据
    joints_2d_list = [d['joints_2d'] for d in filtered_data]
    
    # 转换为 NumPy 数组
    joints_2d_array = np.array(joints_2d_list)
    
    return joints_2d_array

def get_transform(image, pt):
    src_height, src_width, _ = image.shape
    x, y = pt[0], pt[1]
    
    # 计算目标点的坐标（按比例缩放）
    x_ratio = dst_width / src_width
    y_ratio = dst_height / src_height
    dst_point = np.float32([[x * x_ratio, y * y_ratio]]) 

    # 仿射变换的控制点
    src_points = np.float32([[0, 0], [src_width-1, 0], [0, src_height-1]])  
    dst_points = np.float32([[0, 0], [dst_width-1, 0], [0, dst_height-1]])
    
    # 计算仿射变换矩阵
    M = cv2.getAffineTransform(src_points[:3], dst_points[:3])
    
    # 应用仿射变换矩阵到原始点
    transformed_point = np.dot(pt, M)
    
    x1 = int(transformed_point[0]) 
    y1 = int(transformed_point[1]) 
    return x1, y1

def process_images(input_dir, pts_output_dir, resize_output_dir, joints_2d_array):
    # 确保输出目录存在
    if not os.path.exists(pts_output_dir):
        os.makedirs(pts_output_dir)
    if not os.path.exists(resize_output_dir):
        os.makedirs(resize_output_dir)

    # 获取目录下所有图片文件
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    # 用于保存变换后的坐标的字典
    transformed_coords_dict = {}
    coords_dict = {}

    # 遍历每张图片
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片 {image_path}")
            continue
        
        # 生成缩放后的图片
        resized_img = cv2.resize(img, (dst_width, dst_height))  # (w,h)

        # 获取对应的 joints_2d 数据
        if idx >= len(joints_2d_array):
            print(f"没有找到对应的 joints_2d 数据，跳过图片 {image_file}")
            continue
        
        joints_2d = joints_2d_array[idx]

        # 存储当前图片的所有转换后的坐标
        transformed_coords = []
        coords = []

        for pt in joints_2d:
            x = int(pt[0]) - 196
            y = int(pt[1]) - 162

            pt[0] = x   #
            pt[1] = y   #
            
            # 在原图上标注绿色点
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
            
            # 获取转换后的坐标
            x1, y1 = get_transform(img, pt)
            
            # 在缩放后的图上标注蓝色点
            cv2.circle(resized_img, (x1, y1), 1, (255, 0, 0), 2)
            
            coords.append((y,x))
            # 保存转换后的坐标
            transformed_coords.append((y1,x1))

        # 将图片的文件名作为键，转换后的坐标作为值保存到字典中
        transformed_coords_dict[image_file] = transformed_coords
        coords_dict[image_file] = coords

        # 保存标注后的原图到 pts_output_dir 目录
        output_pts_image_path = os.path.join(pts_output_dir, image_file)
        cv2.imwrite(output_pts_image_path, img)
        
        # 保存缩放后的图片到 resize_output_dir 目录
        output_resize_image_path = os.path.join(resize_output_dir, image_file.replace('.jpg', '_tmp.jpg'))
        cv2.imwrite(output_resize_image_path, resized_img)
        
        print(f"处理并保存图片 {output_pts_image_path} 和缩放图片 {output_resize_image_path}")

    # 保存变换后的坐标字典为.pkl文件
    with open(os.path.join(pkl_dir, 's_05_act_02_subact_02_ca_03_transformed_coords_cropped.pkl'), 'wb') as f:
        pickle.dump(transformed_coords_dict, f)
    
    with open(os.path.join(pkl_dir, 's_05_act_02_subact_02_ca_03_coords_cropped.pkl'), 'wb') as f:
        pickle.dump(coords_dict, f)
    
    print(f"变换后的坐标已保存到 {os.path.join(pts_output_dir, 'transformed_coords_cropped.pkl')}")

if __name__ == '__main__':
    input_dir = "/home/tanli/Code/dataset/H36M_set/h36m_cropped/s_05_act_02_subact_02_ca_03"
    pkl_dir = "/home/tanli/Code/dataset/H36M_set/cropped_coords_pkl"
    pts_output_dir = "/home/tanli/Code/dataset/H36M_set/h36m_pts_cropped/s_05_act_02_subact_02_ca_03"
    resize_output_dir = "/home/tanli/Code/dataset/H36M_set/h36m_resize_cropped/s_05_act_02_subact_02_ca_03"
    pkl_file = "/media/qcitStore/tanli/Code/dataset/H36M_set/h36m_train.pkl"

    joints_2d_array = extract_joints_2d_from_pkl(pkl_file)

    # 处理图片并保存转换后的坐标
    process_images(input_dir, pts_output_dir, resize_output_dir, joints_2d_array)

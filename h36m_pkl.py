import pickle  
import numpy as np
  
# 加载pickle文件  
#with open('/home/tanli/Code/dataset/H36M_set/cropped_coords_pkl/s_01_act_02_subact_02_ca_01_transformed_coords_cropped.pkl', 'rb') as f:  
with open('/home/tanli/Code/dataset/H36M_set/h36m_train.pkl', 'rb') as f:  
    data = pickle.load(f)  

# 假设data是一个列表，其中每个元素都是一个字典  
filtered_data = [d for d in data if d['action'] == 2 and d['subaction'] == 2 and d['camera_id'] == 2 and d['subject'] == 5 ]  
box_dict = {}

for item in filtered_data:
    # 提取 image 和 joints_3d
    image = item.get('image', 'N/A')
    box = item.get('box', None)
    
    if box is not None:
        box_dict[image] = box  # 将 joints_3d 数据存入字典
    
print(box_dict)

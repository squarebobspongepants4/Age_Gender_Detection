# import os
# import torch
# import cv2
# # import json
# import numpy as np
# import pandas as pd
# import time

# from torch import nn
# from google.colab.patches import cv2_imshow

# #video frame capturer
# def FrameCapture(video_name):
#   vidObj = cv2.VideoCapture('/REPO/data/videos/'+video_name)
#   count = 0
#   success = 1
#   while success:
#       success, image = vidObj.read()
#       cv2.imwrite("/REPO/data/images/frame%d.jpg" % count, image)
#       count += 1


# #utils
# df=pd.DataFrame(columns=['image','person_id','bbox','age','gender'])
# path_to_file='/REPO/output/bbox_file.txt'
# !touch {path_to_file}
# big_image_list=[]
# big_face_id=[]
# big_bbox_list=[]
# big_age_list=[]
# big_gender_list=[]

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Gender=['Female','Male']
# age_class=['0-10','10-20','20-30','30-40','40-50','50-60','60=70','70-80','80-90','90-100']
# image_size=128
# test_transforms = transforms.Compose([transforms.ToPILImage(),
#                                      transforms.Resize((image_size,image_size)),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.5,0.5,0.5],
#                                                           [0.5,0.5,0.5])])


# ##loading yolo weights
# #options available : 5l-face.pt,5m-face,5s-face.pt
# weight='5l-face.pt'


# #loading age-gender detector
# from torchvision.models import resnet50
# gender_model=resnet50(pretrained=True)
# age_model=resnet50(pretrained=True)

# gender_model.fc = nn.Sequential(
#     nn.Linear(2048, 100),
#     nn.ReLU(),
#     nn.Linear(100, 2),
# )
# age_model.fc = nn.Sequential(
#     nn.Linear(2048, 100),
#     nn.ReLU(),
#     nn.Linear(100, 10),
# )
# gender_model_path='/REPO/weights/gender.pth.tar'
# age_model_path='/REPO/weights/age.pth.tar'

# if device =='cpu':
#   print("preparing to run on CPU")
#   gender_model.load_state_dict(torch.load(gender_model_path,map_location=torch.device('cpu'))['model_state_dict'])
#   age_model.load_state_dict(torch.load(age_model_path,map_location=torch.device('cpu'))['model_state_dict'])
# else:
#   print("preparing to run on GPU")
#   gender_model.load_state_dict(torch.load(gender_model_path)['model_state_dict'])
#   age_model.load_state_dict(torch.load(age_model_path)['model_state_dict'])

# root_data_path='/REPO/data'
# dataset_path=root_data_path+'/images' ## set path of to-be-tested images
# temp_storage=root_data_path+'/temp'
# cropped_faces_path=temp_storage+'/faces'
# _,_,target_image_files=next(os.walk(dataset_path))

# #create a text file to write bboxes
# for image_file in target_image_files:#takes 1 image
#   !mkdir {cropped_faces_path}
#   print("################################################################################")
#   print("WORKING ON IMAGE: " ,image_file)
#   original_image_file_path=dataset_path+'/'+image_file
#   before_image=cv2.imread(original_image_file_path)
  
#   print("BEFORE ENHANCING IMAGE:")
#   cv2_imshow(before_image)

#   #apply esr-gan
#   %cd /REPO/Real-ESRGAN
#   upsampled_path=temp_storage
#   ## if gpu runs out of memory enable the --tile argument in the next line
#   !python inference_realesrgan.py -n /REPO/weights/RealESRGAN_x4plus.pth -i {original_image_file_path} -o {upsampled_path} --outscale 3.5  --face_enhance #--tile 1000  
#   upsampled_image_path=temp_storage+'/'+'sr.jpg'
#   upsampled_image=cv2.imread(upsampled_image_path)
#   print("AFTER ENHANCING IMAGE:")
#   cv2_imshow(upsampled_image)
#   # break

#   # face detector
#   %cd /REPO/yolov5-face/
#   !python detect.py --weights /REPO/weights/{weight} --image {upsampled_image_path} 
  
#   #iterate over each cropped face
#   _,_,cropped_faces =next(os.walk(cropped_faces_path))
#   face_id=0
#   for cropped_face in cropped_faces:
#     a=time.time()
#     face_path=cropped_faces_path+'/'+cropped_face
#     face=cv2.imread(face_path)
#     print("FACE: ",cropped_face)
#     cv2_imshow(face)

#     ## pass on age-gender model
#     face = test_transforms(face)
#     face = face.unsqueeze(0)
#     gender_output = gender_model(face)
#     age_output = age_model(face)
#     gender_pred = gender_output.cpu().data.max(1, keepdim=True)[1]
#     age_pred = age_output.cpu().data.max(1, keepdim=True)[1]
#     # print(gender_output)
#     # print(age_output)
#     gender = gender_pred.detach().cpu().numpy().item()
#     age = age_class[age_pred.detach().cpu().numpy().item()]
#     b=time.time()
#     print("GENDER: ",gender," AGE: ",age)
#     print('ESTIMATED IN ',b-a," secs")
#     print('----------')

#     big_image_list.append(image_file)
#     big_face_id.append(face_id)
#     face_id+=1
#     big_age_list.append(age)
#     big_gender_list.append(Gender[gender])
#   !rm -r {cropped_faces_path}
#   # break

# #write in df
# df['image']=big_image_list
# df['person_id']=big_face_id
# df['age']=big_age_list
# df['gender']=big_gender_list

# with open(path_to_file) as topo_file:
#     for line in topo_file:
#       big_bbox_list.append(line)

# ##saving as csv
# df.to_csv('/REPO/output/output.csv')
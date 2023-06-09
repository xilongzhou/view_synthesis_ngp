
import os
from os.path import join as join

import numpy as np

import torch
from torch.utils.data import DataLoader


import torch.utils.data as data

from PIL import Image
import random

class DataLoader_helper(data.Dataset):
	"""docstring for  DataLoader_helper"""
	def __init__(self, path, h_res, w_res, row=12, col=12):
		super( DataLoader_helper, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res
		self.row = row
		self.col = col

		# id of left and right stereo input
		self.all_ids = [i for i in range(self.row)]
		print("self.all_ids: ", self.all_ids)
		self.L_id = 0
		self.R_id = 11
		# self.all_ids.remove(self.L_id)
		# self.all_ids.remove(self.R_id)
		print("self.all_ids: ", self.all_ids)

		# compute interpolated
		dist = 1/(self.R_id - self.L_id)
		self.inter_list = [np.float32(i*dist) for i in range(self.R_id - self.L_id+1)]
		print("self.inter_list: ", self.inter_list)

		self.image_filenames = [x for x in os.listdir(self.path)]
		self.num_all = len(self.image_filenames)

		self.total_length = int(self.num_all/self.row)
		print("self.total_length : ", self.total_length )

		assert self.num_all % self.row == 0, "total number not match images per row"
		assert len(self.all_ids) == len(self.inter_list), "the length of IDs and interlist not match"

	def __getitem__(self, index):

		# fetch image based on ID
		scene_id = index // self.col

		row_id = index % self.col

		imgL_id = row_id*self.col + self.L_id
		imgR_id = row_id*self.col + self.R_id
		rand_id = random.choice(self.all_ids) 
		imgtest_id = row_id*self.col + rand_id

		# reconstruct image name
		imgL_name = f"idx{scene_id}_{imgL_id:05d}.png"
		imgR_name = f"idx{scene_id}_{imgR_id:05d}.png"
		imgtest_name = f"idx{scene_id}_{imgtest_id:05d}.png"

		# load image per row
		imgL_pil = Image.open(join(self.path, imgL_name)).convert("RGB")
		imgR_pil = Image.open(join(self.path, imgR_name)).convert("RGB")
		imgtest_pil = Image.open(join(self.path, imgtest_name)).convert("RGB")

		# resize if needed:
		imgL_pil = imgL_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		imgR_pil = imgR_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		imgtest_pil = imgtest_pil.resize((self.w_res, self.h_res), Image.LANCZOS )


		imgL = torch.from_numpy(np.array(imgL_pil)).permute(2,0,1)/255.
		imgR = torch.from_numpy(np.array(imgR_pil)).permute(2,0,1)/255.
		imgtest = torch.from_numpy(np.array(imgtest_pil)).permute(2,0,1)/255.

		# interpolated val
		inter_val = self.inter_list[rand_id]

		# print(f"index: {index}, imgL: {imgL_name}, imgR: {imgR_name}, imgtest: {imgtest_name}")

		return imgL, imgR, imgtest, inter_val, index

	def __len__(self):
		return self.total_length

	def get_total_num(self):
		return self.total_length


# load mask and all images
class DataLoader_helper2(data.Dataset):
	"""docstring for  DataLoader_helper"""
	def __init__(self, path, h_res, w_res, num=12, plane=6, one_scene=False, load_model=False):
		super( DataLoader_helper2, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res
		self.num = num
		self.plane = plane
		self.load_model = load_model

		self.all_scenes = os.listdir(path)
		self.total_length = len(self.all_scenes) if not one_scene else 1

		print("length of data ", self.total_length)

		# id of left and right stereo input
		self.all_ids = [i for i in range(self.num)]
		print("self.all_ids: ", self.all_ids)
		self.L_id = 0
		self.R_id = 11
		print("self.all_ids: ", self.all_ids)

		# compute interpolated
		dist = 1/(self.R_id - self.L_id)
		self.inter_list = [np.float32(i*dist) for i in range(self.R_id - self.L_id+1)]
		print("self.inter_list: ", self.inter_list)

	def __getitem__(self, index):

		path = join(self.path, self.all_scenes[index])

		rand_id = random.choice(self.all_ids) 

		# load image 
		imgL_name = f"rgba_00000.png"
		imgR_name = f"rgba_00011.png"
		imgtest_name = f"rgba_{rand_id:05d}.png"

		imgL_pil = Image.open(join(path, imgL_name)).convert("RGB")
		imgR_pil = Image.open(join(path, imgR_name)).convert("RGB")
		imgtest_pil = Image.open(join(path, imgtest_name)).convert("RGB")

		# # resize if needed:
		# imgL_pil = imgL_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		# imgR_pil = imgR_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		# imgtest_pil = imgtest_pil.resize((self.w_res, self.h_res), Image.LANCZOS )

		imgL = torch.from_numpy(np.array(imgL_pil)).permute(2,0,1)/255.
		imgR = torch.from_numpy(np.array(imgR_pil)).permute(2,0,1)/255.
		imgtest = torch.from_numpy(np.array(imgtest_pil)).permute(2,0,1)/255.

		# load mask
		all_maskL = []
		all_maskR = []
		for i in range(self.plane):
			
			# left
			maskL_name = f"mask{i}_00000.png"
			maskL = np.array(Image.open(join(path, maskL_name)))
			maskL = torch.from_numpy(maskL)/255.

			# right
			maskR_name = f"mask{i}_00011.png"
			maskR = np.array(Image.open(join(path, maskR_name)))
			maskR = torch.from_numpy(maskR)/255.	

			all_maskL.append(maskL)		
			all_maskR.append(maskR)		

		all_maskL = torch.stack(all_maskL)
		all_maskR = torch.stack(all_maskR)

		# load disparity
		disparity_0_11 = np.load(os.path.join(path, "disp_0_11.npy"))

		# interpolated val
		inter_val = self.inter_list[rand_id]

		# load model
		if self.load_model:
			model_path = join(path, "model.ckpt")
			return imgL, imgR, imgtest, inter_val, index, all_maskL, all_maskR, disparity_0_11, model_path

		else:
			return imgL, imgR, imgtest, inter_val, index, all_maskL, all_maskR, disparity_0_11

	def __len__(self):
		return self.total_length

	def get_total_num(self):
		return self.total_length


# load mask and all images
class DataLoader_helper_test(data.Dataset):
	"""docstring for  DataLoader_helper"""
	def __init__(self, path, h_res, w_res, num=12, plane=6, one_scene=True):
		super( DataLoader_helper_test, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res
		self.num = num
		self.plane = plane

		self.scene = path
		self.total_length = 1

		print("length of data ", self.total_length)

		# id of left and right stereo input
		self.all_ids = [i for i in range(self.num)]
		print("self.all_ids: ", self.all_ids)
		self.L_id = 0
		self.R_id = 11
		print("self.all_ids: ", self.all_ids)

		# compute interpolated
		dist = 1/(self.R_id - self.L_id)
		self.inter_list = [np.float32(i*dist) for i in range(self.R_id - self.L_id+1)]
		print("self.inter_list: ", self.inter_list)

	def __getitem__(self, index):

		# load disparity
		disparity_0_11 = np.load(os.path.join(self.scene, "disp_0_11.npy"))

		return  disparity_0_11

	def __len__(self):
		return self.total_length



class DataLoader_helper_blend(data.Dataset):
	"""docstring for  DataLoader_helper"""
	def __init__(self, path, h_res, w_res, num=12, plane=6,):
		super( DataLoader_helper_blend, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res
		self.num = num
		self.plane = plane

		self.all_scenes = os.listdir(path)
		self.total_length = len(self.all_scenes)

		print("length of data ", self.total_length)

		# id of left and right stereo input
		self.all_ids = [i for i in range(self.num)]
		print("self.all_ids: ", self.all_ids)
		self.L_id = 0
		self.R_id = 11
		print("self.all_ids: ", self.all_ids)

		# compute interpolated
		dist = 1/(self.R_id - self.L_id)
		self.inter_list = [np.float32(i*dist) for i in range(self.R_id - self.L_id+1)]
		print("self.inter_list: ", self.inter_list)

	def __getitem__(self, index):

		path = join(self.path, self.all_scenes[index])

		rand_id = random.choice(self.all_ids) 

		# load image 
		imgtest_name = f"rgba_{rand_id:05d}.png"

		imgtest_pil = Image.open(join(path, imgtest_name)).convert("RGB")

		# # resize if needed:
		# imgL_pil = imgL_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		# imgR_pil = imgR_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		# imgtest_pil = imgtest_pil.resize((self.w_res, self.h_res), Image.LANCZOS )

		imgtest = torch.from_numpy(np.array(imgtest_pil)).permute(2,0,1)/255.

		# load disparity
		disparity_0_11 = np.load(os.path.join(path, "disp_0_11.npy"))
		# print("disparity_0_11: ", disparity_0_11.shape)

		disparity_0_11 = np.absolute(disparity_0_11[:,:,0])
		disp_max = np.amax(disparity_0_11)
		disp_min = np.amin(disparity_0_11)

		# print(path, " disp_max: ", disp_max)

		# interpolated val
		inter_val = self.inter_list[rand_id]

		# load model
		model_path = join(path, "model.ckpt")
		return imgtest, inter_val, index, disp_max, disp_min, model_path


	def __len__(self):
		return self.total_length

	def get_total_num(self):
		return self.total_length



class DataLoader_helper_blend_sep(data.Dataset):
	"""docstring for  DataLoader_helper"""
	def __init__(self, path, h_res, w_res, num=12, plane=6,):
		super( DataLoader_helper_blend_sep, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res
		self.num = num
		self.plane = plane

		self.all_scenes = os.listdir(path)
		self.total_length = len(self.all_scenes)

		print("length of data ", self.total_length)

		# id of left and right stereo input
		self.all_ids = [i for i in range(self.num)]
		print("self.all_ids: ", self.all_ids)
		self.L_id = 0
		self.R_id = 11
		print("self.all_ids: ", self.all_ids)

		# compute interpolated
		dist = 1/(self.R_id - self.L_id)
		self.inter_list = [np.float32(i*dist) for i in range(self.R_id - self.L_id+1)]
		print("self.inter_list: ", self.inter_list)

	def __getitem__(self, index):

		path = join(self.path, self.all_scenes[index])

		rand_id = random.choice(self.all_ids) 

		# load image 
		imgtest_name = f"rgba_{rand_id:05d}.png"

		imgtest_pil = Image.open(join(path, imgtest_name)).convert("RGB")

		# # resize if needed:
		# imgL_pil = imgL_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		# imgR_pil = imgR_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		# imgtest_pil = imgtest_pil.resize((self.w_res, self.h_res), Image.LANCZOS )

		imgtest = torch.from_numpy(np.array(imgtest_pil)).permute(2,0,1)/255.

		# load disparity
		disparity_0_11 = np.load(os.path.join(path, "disp_0_11.npy"))
		# print("disparity_0_11: ", disparity_0_11.shape)

		disparity_0_11 = np.absolute(disparity_0_11[:,:,0])
		disp_max = np.amax(disparity_0_11)
		disp_min = np.amin(disparity_0_11)

		# print(path, " disp_max: ", disp_max)

		# interpolated val
		inter_val = self.inter_list[rand_id]

		# load model
		model_path = join(path, "model")
		return imgtest, inter_val, index, disp_max, disp_min, model_path


	def __len__(self):
		return self.total_length

	def get_total_num(self):
		return self.total_length


class DataLoader_helper_blend_test(data.Dataset):
	"""docstring for  DataLoader_helper_blend_test"""
	def __init__(self, path, h_res, w_res, num=12, plane=6):
		super( DataLoader_helper_blend_test, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res
		self.plane = plane

		self.all_scenes = os.listdir(path)
		self.total_length = len(self.all_scenes)

	def __getitem__(self, index):

		path = join(self.path, self.all_scenes[index])

		# load disparity
		disparity_0_11 = np.load(os.path.join(path, "disp_0_11.npy"))
		# print("disparity_0_11: ", disparity_0_11.shape)

		disparity_0_11 = np.absolute(disparity_0_11[:,:,0])
		disp_max = np.amax(disparity_0_11)
		disp_min = np.amin(disparity_0_11)

		# load model
		model_path = join(path, "model.ckpt")

		return  disp_max, disp_min, model_path


	def __len__(self):
		return self.total_length

	def get_total_num(self):
		return self.total_length




class DataLoader_helper_test2(data.Dataset):
	"""docstring for  DataLoader_helper"""
	def __init__(self, path, h_res, w_res):
		super( DataLoader_helper_test2, self).__init__()
		self.path = path
		self.h_res = h_res
		self.w_res = w_res

		self.image_allfilenames = [x.split('.')[0][:-2] for x in os.listdir(self.path)]

		self.image_filenames=[]
		for name in self.image_allfilenames:
			if name not in self.image_filenames:
				self.image_filenames.append(name)

		print("image_filenames: ", self.image_filenames)


	def __getitem__(self, index):

		# fetch image based on ID
		name = self.image_filenames[index]

		# reconstruct image name
		imgL_name = f"{name}_0.png"
		imgR_name = f"{name}_1.png"

		# load image per row
		imgL_pil = Image.open(join(self.path, imgL_name)).convert("RGB")
		imgR_pil = Image.open(join(self.path, imgR_name)).convert("RGB")

		# resize if needed:
		imgL_pil = imgL_pil.resize((self.w_res, self.h_res), Image.LANCZOS )
		imgR_pil = imgR_pil.resize((self.w_res, self.h_res), Image.LANCZOS )


		imgL = torch.from_numpy(np.array(imgL_pil)).permute(2,0,1)/255.
		imgR = torch.from_numpy(np.array(imgR_pil)).permute(2,0,1)/255.

		return imgL, imgR, name

	def __len__(self):
		return len(self.image_filenames)



if __name__ =="__main__":

	path = "D:/XilongZhou/Research/Research_Meta/Dataset/SynData/SynData_s1_all"

	Myset = DataLoader_helper(path)
	Mydata = DataLoader(dataset=Myset, batch_size=2, shuffle=True, drop_last=True)

	for i,data in enumerate(Mydata):
		# print(len(data))
		imgL, imgR, imgtest, inter_val = data
		print("imgL: ",imgL.shape)
		print("imgR: ",imgR.shape)
		print("imgtest: ",imgtest.shape)
		print("inter_val: ",inter_val)

		for k in range(2):
			imgL_t = imgL[k]
			print("imgL_t: ",imgL_t.shape)

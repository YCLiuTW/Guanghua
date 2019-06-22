import os,json
import numpy as np
import torch
import torchvision.transforms as tr
from torch.utils import data
from PIL import Image

main_path = os.getcwd()

class CatDog_Loader(data.Dataset):

	def __init__(self,data_root,train,crop_size = (32,32)):
		self.kind = ['cat','dog']
		self.crop_size = crop_size

		if train:
			self.data_path = os.path.join(main_path,data_root,'train')
		else:
			self.data_path = os.path.join(main_path,data_root,'test')
		
		self.pre_load()

	def __len__(self):
		return len(self.files)

	def __getitem__(self,index):
		
		file = self.files[index]

		label = file['label']

		img = Image.open(file['img_path'])
		img = img.resize(self.crop_size,Image.BICUBIC)
		img = np.asarray(img,np.float32)
		img = img.transpose((2, 0, 1))

		return img,label

	def pre_load(self):
		#container for data
		self.files = []

		for index,kind in enumerate(self.kind):
			current_path = os.path.join(self.data_path,kind)
			
			for data in os.listdir(current_path):
				if data.endswith('.jpg'):
					self.files.append({
						'img_path' : os.path.join(current_path,data),
						'label': index
					})

if __name__ == "__main__":
	# Data Loader Unit Test
	dst = CatDog_Loader(data_root = 'data',train = True)
	train_dataloader = data.DataLoader(dataset = dst, batch_size = 10, shuffle = True)
	for i,data in enumerate(train_dataloader,start = 1):
		img,label = data
		print(img.shape)
		print(label)
		break
	print("Test OK!!")




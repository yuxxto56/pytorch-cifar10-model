"""
@File  main.py
@Author 65451<654516092@qq.com>
@Date 2023/4/20 14:12
"""
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img = Image.open("./img.png", "r")
transforms = torchvision.transforms.Compose([
	torchvision.transforms.RandomCrop(214, padding=4),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.ToTensor(),
	#torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
img_origin_data = np.array(img)

img_tensor = transforms(img)
write = SummaryWriter("./logs")
write.add_image(tag="single_img_main", img_tensor=img_origin_data, global_step=1, dataformats="HWC")
write.add_image(tag="single_img_main", img_tensor=img_tensor, global_step=2)
write.flush()
write.close()

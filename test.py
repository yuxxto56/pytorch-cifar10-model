"""
@File  test.py
@Author 65451<654516092@qq.com>
@Date 2023/4/28 10:26
"""
import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from Model.Resnet import ResNet50


def pridiect(model, img_path):
	img = Image.open(img_path)
	transform_test = transforms.Compose([
		transforms.Resize(size=(32, 32)),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	img = transform_test(img)
	img_predict = torch.reshape(img, shape=(1, 3, img.shape[1], img.shape[2]))
	model.eval()
	with torch.no_grad():
		output = model(img_predict)
		classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
		print("{}|图片预测结果是：{}".format(img_path, classes[output.argmax().item()]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pridicting')
	parser.add_argument('--img_dir', type=str, default="TestImg")
	opt = parser.parse_args()
	if os.path.isdir(opt.img_dir) is False:
		print("{} is not exists.".format(opt.img_dir))
		exit()

	list_imgs = [opt.img_dir + "/{}".format(img) for img in os.listdir(opt.img_dir)]
	model = ResNet50()
	model.load_state_dict(torch.load("./Res/ResNet50-CIFAR10.pth", map_location="cpu")["net"])
	for path in list_imgs:
		pridiect(model, path)

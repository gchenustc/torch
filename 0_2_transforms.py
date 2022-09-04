from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter("log")

image_path = "./dataset/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)

# --- ToTensor ---
# 创建 ToTensor 实例
tensor_trans = transforms.ToTensor()
# 转换格式 - tensor
img_tensor = tensor_trans(img_PIL) # img_tensor元素的值介于0-1之间, 并且是CHW格式
# 转换格式 - np
img_numpy = np.array(img_PIL) # img_numpy的值介于0-255之间，并且是HWC格式

# print(img_tensor, img_tensor.shape)
# print(img_numpy, img_numpy.shape) # 

writer.add_image("img_tensor",img_tensor,1)
writer.add_image("image_np",img_numpy,1,dataformats="HWC")

# --- Normalize ---
# (VALUE - MEAN)/STD
tensor_normalize = transforms.Normalize(mean = [0.5,0.5,0.5], std =[0.5,0.5,0.5])  # 列表三元素三个通道
img_tensor_norm = tensor_normalize(img_tensor)
# img_numpy_norm = tensor_normalize(img_numpy)  # 只有tensor类型可以调用
# print(img_tensor[0][0][0:3], img_tensor_norm[0][0][0:3])

writer.add_image("img_tensor_norm", img_tensor_norm, 1)

# print(img_tensor.shape)


# --- Resize ---
tensor_resize = transforms.Resize(size=(50,50)) 
img_tensor_resize = tensor_resize(img_tensor) # 只能传入tensor类型
writer.add_image("img_tensor_resize", img_tensor_resize, 1)


# --- Compose ---
# 将几种transform合并
trans_totensor = transforms.ToTensor()
trans_norm = transforms.Normalize([1,2,3],[3,2,1])
trans_resize = transforms.Resize((500,500))
trans_compose = transforms.Compose([trans_totensor, trans_norm, trans_resize])
img_compose = trans_compose(img_PIL)
writer.add_image("img_compose", img_compose, 1)


# --- RandomCrop ---
# 随即裁剪
trans_crop = transforms.RandomCrop((40,40))
for i in range(100):
    img = trans_crop(img_tensor)
    writer.add_image("img_randomcrop", img, i)

writer.close()
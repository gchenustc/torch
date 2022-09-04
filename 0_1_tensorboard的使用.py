from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter

# 初始化SummaryWriter
writer = SummaryWriter("logs") # tensorboard存放文件的文件夹名为logs

# writer.add_scalar() - 横坐标步数，纵坐标值的图形展示
for i in range(100):
    writer.add_scalar(tag="y=2x", scalar_value=2*i, global_step=i)

# 添加图片
from PIL import Image
image_path = "./dataset/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path) # PIL格式的图片
import numpy as np
img_np = np.array(img_PIL)

# HWC指的是图片格式是(高，宽，通道)，默认是(通道，高，宽)
writer.add_image(tag="img1",img_tensor=img_np,global_step=1,dataformats="HWC") # 图片格式要求：torch.Tensor, numpy.array, or string/blobname，所以要将PIL转换成numpy/tensor格式
writer.add_image(tag="img1",img_tensor=img_np,global_step=2,dataformats="HWC") # 图片格式要求：torch.Tensor, numpy.array, or string/blobname，所以要将PIL转换成numpy/tensor格式

# 使用结束后要关闭
# 如果展示内容有变化，改动代码后运行，tensorboard的内容会和原有的图表重叠，可以直接删掉缓存文件重新运行
writer.close()
# 在终端打开tensorboard的命令 tensorboard --logdir=logs --port 6006



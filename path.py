# path.py

# --- 扩散模型预训练权重路径 ---
# 请确保路径指向具体的 .pt 权重文件
diffusion_model_path = {
    'cifar10': '/SSD_Data01/HHY/cifar10/checkpoint_8.pth',
    # 【修改】指向您存放ImageNet扩散模型权重的完整路径
    'imagenet': '/SSD_Data01/HHY/imagenet/256x256_diffusion_uncond.pt',
    'svhn': ''
}

cifar10_clf_path = '/SSD_Data01/HHY/wide_resnet/weights-best.pt'
imagenet_clf_path = '/SSD_Data01/HHY/resnet/resnet50-19c8e357.pth'

# --- 分类器预训练权重路径 ---
# (如果项目中有用到SVHN分类器的独立权重)
svhn_clf_path = ''

# --- 数据集根目录路径 ---
# 请确保路径指向存放数据集压缩包的文件夹
# torchvision会自动在此目录下寻找并解压数据文件
# 【修改】指向您存放ImageNet数据集的文件夹
imagenet_path = '/SSD_Data01/HHY'

# --- CIFAR-10 和 SVHN 数据集路径 ---
# 通常，CIFAR-10和SVHN会自动下载到项目根目录下的dataset文件夹中
# 如果您有自定义的存放位置，也可以在这里指定
# cifar10_path = './dataset' 
# svhn_path = './dataset'
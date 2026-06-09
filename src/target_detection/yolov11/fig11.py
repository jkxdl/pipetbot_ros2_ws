import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math

# 全局字典，用于保存 hook 捕获的特征图
feature_maps = {}

def get_hook(name):
    """返回 hook 函数，将对应层的输出保存到 feature_maps 字典中"""
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu().numpy()
    return hook

def register_hooks(model):
    """
    利用 named_modules 遍历所有子模块，为每个卷积层注册 forward hook。
    如果 model.model 不支持 named_modules，则尝试使用 model.model.model
    """
    hooks = []
    if hasattr(model.model, "named_modules"):
        actual_model = model.model
    elif hasattr(model.model, "model") and hasattr(model.model.model, "named_modules"):
        actual_model = model.model.model
    else:
        raise AttributeError("无法找到模型内部的 torch 模型结构！")
        
    for name, module in actual_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            h = module.register_forward_hook(get_hook(name))
            hooks.append(h)
    return hooks

def tile_images(img_list, cols=4, pad=2):
    """
    将多张图片按网格叠放在一起，返回叠放后的大图。
    参数：
      img_list: 每个元素为单通道图像（二维数组），尺寸需一致
      cols: 每行显示图片数
      pad: 图片之间的间隙（像素）
    """
    if len(img_list) == 0:
        return None
    h, w = img_list[0].shape
    rows = math.ceil(len(img_list) / cols)
    tiled_img = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad), dtype=np.uint8)
    tiled_img.fill(0)
    for idx, img in enumerate(img_list):
        r = idx // cols
        c = idx % cols
        start_y = r * (h + pad)
        start_x = c * (w + pad)
        tiled_img[start_y:start_y+h, start_x:start_x+w] = img
    return tiled_img

def process_feature_map(feat):
    """
    针对不同维度的特征图进行处理，返回每个通道归一化后的图像列表。
    对于每个通道，根据该通道的最小值和最大值分别归一化。
    如果最大值和最小值相同，则生成一张提示性图像。
    """
    imgs = []
    if feat.ndim == 4:
        # 假设 feat 的 shape 为 (B, C, H, W)，取第一个样本
        B, C, H, W = feat.shape
        for ch in range(C):
            channel_img = feat[0, ch, :, :]
            ch_min, ch_max = channel_img.min(), channel_img.max()
            print(f"Channel {ch}: min={ch_min}, max={ch_max}")
            if abs(ch_max - ch_min) < 1e-6:
                # 数值完全一致，生成一张全白图或全灰图，方便观察
                norm_img = np.full(channel_img.shape, 255, dtype=np.uint8)
                # 你也可以选择生成全0图：np.zeros(channel_img.shape, dtype=np.uint8)
            else:
                norm_img = ((channel_img - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
            imgs.append(norm_img)
    elif feat.ndim == 2:
        # 假设 feat 的 shape 为 (C, flat)
        C, flat_dim = feat.shape
        H = int(np.sqrt(flat_dim))
        while H > 0:
            if flat_dim % H == 0:
                W = flat_dim // H
                break
            H -= 1
        else:
            print(f"无法重构特征图，flat_dim={flat_dim}")
            return imgs
        for ch in range(C):
            channel_img = feat[ch, :].reshape(H, W)
            ch_min, ch_max = channel_img.min(), channel_img.max()
            print(f"Channel {ch}: min={ch_min}, max={ch_max}")
            if abs(ch_max - ch_min) < 1e-6:
                norm_img = np.full(channel_img.shape, 255, dtype=np.uint8)
            else:
                norm_img = ((channel_img - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
            imgs.append(norm_img)
    else:
        print(f"不支持的特征图维度: {feat.shape}")
    return imgs


def main():
    # 修改模型路径为你的模型文件路径
    model_path = '/home/robot/aubo_ros2_ws/dataset/runs/train/yolov11n-HSFPN_experiment/weights/best.pt'
    model = YOLO(model_path)
    
    # 注册 hook 捕获所有卷积层输出
    hooks = register_hooks(model)
    
    # 修改图片路径为你的测试图片路径
    image_path = '/home/robot/aubo_ros2_ws/dataset/YOLO_dataset/Train/images/TestTubeRack053_png.rf.9b5807fd7259ad1a078de6f6a8f07aa8.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("加载图片失败，请检查图片路径")
        return
    # 转换 BGR 为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 前向传播，触发 hook
    _ = model(image_rgb)
    
    # 取消 hook 注册
    for h in hooks:
        h.remove()
    
    # 分别处理并显示每个捕获到的特征图
    num_layers = len(feature_maps)
    plt.figure(figsize=(15, 15))
    idx_plot = 1
    for name, feat in feature_maps.items():
        channel_imgs = process_feature_map(feat)
        if not channel_imgs:
            continue
        # 如果只有一个通道，直接展示；否则，使用 tile_images 将所有通道图像拼接在一起
        if len(channel_imgs) == 1:
            tiled = channel_imgs[0]
        else:
            tiled = tile_images(channel_imgs, cols=4, pad=2)
        plt.subplot(math.ceil(num_layers/2), 2, idx_plot)
        plt.imshow(tiled, cmap='viridis')
        plt.title(f"{name}\nChannels: {len(channel_imgs)}")
        plt.axis('off')
        idx_plot += 1
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

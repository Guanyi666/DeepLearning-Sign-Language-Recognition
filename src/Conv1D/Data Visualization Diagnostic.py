import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_features(file_path, title):
    """
    加载一个 .npy 文件并将其特征随时间可视化。
    """
    try:
        # 使用你原始代码中的 Path 对象
        p = Path(file_path)
        if not p.exists():
            print(f"!!! 错误: 找不到文件 {file_path}")
            print("请确保下面的路径是正确的！")
            return
            
        data = np.load(p).astype(np.float32) # 加载 (30, 126)
        
        plt.figure(figsize=(15, 6))
        
        if np.all(data == 0):
            # ！！！这是最可能的“病因”！！！
            plt.text(0.5, 0.5, '数据文件全是 0！(Data is all Zeros!)', 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=20, color='red', transform=plt.gca().transAxes)
            print(f"诊断: {title} ({p.name}) 的数据全是 0！")
        else:
            # data.T 是 (126, 30)，我们把126个特征随30帧的变化画出来
            plt.plot(data.T) 
            print(f"诊断: {title} ({p.name}) 数据已加载并绘制。")
            
        plt.title(f"特征可视化: {title} ({p.name})", fontsize=16)
        plt.xlabel("帧 (时间)", fontsize=12)
        plt.ylabel("归一化的特征值", fontsize=12)
        plt.ylim(-0.1, 1.1) # 假设你的坐标是 0-1 归一化
        
    except Exception as e:
        print(f"加载或绘制文件 {file_path} 出错: {e}")

# --- ！！！请你修改这里的路径！！！ ---

# 这是你的数据根目录
BASE_DIR = r"D:\code\deeplearning\project\DeepLearning-Sign-Language-Recognition\data\dataset\train"

# 1. 检查 "morning" (失败的类)
# (我们假设 "morning/01/" 目录下有一个 "morning_1_features.npy")
file_to_check_A = os.path.join(BASE_DIR, "morning", "01", "morning_1_features.npy")

# 2. 检查 "very" (失败的类)
file_to_check_B = os.path.join(BASE_DIR, "very", "01", "very_1_features.npy")

# 3. 检查 "good" (被混淆的类)
file_to_check_C = os.path.join(BASE_DIR, "good", "01", "good_1_features.npy")

# 4. 检查 "thank_you" (模型现在疯狂猜测的类)
file_to_check_D = os.path.join(BASE_DIR, "thank_you", "01", "thank_you_1_features.npy")


# --- 运行诊断 ---
# (确保你的 matplotlib 窗口可以弹出)

print("--- 正在诊断 'morning' (失败的类) ---")
visualize_features(file_to_check_A, "Morning (A)")

print("\n--- 正在诊断 'very' (失败的类) ---")
visualize_features(file_to_check_B, "Very (B)")

print("\n--- 正在诊断 'good' (被混淆的类) ---")
visualize_features(file_to_check_C, "Good (C)")

print("\n--- 正在诊断 'thank_you' (偏见的类) ---")
visualize_features(file_to_check_D, "Thank You (D)")

# 最后显示所有图像
plt.show()
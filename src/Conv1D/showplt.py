import matplotlib.pyplot as plt
import numpy as np

def visualize_features(file_path, title):
    """
    加载一个 .npy 文件并将其特征随时间可视化。
    """
    try:
        data = np.load(file_path) # 加载 (30, 126)
        
        # 检查数据是否有效
        if np.all(data == 0):
            print(f"警告: {file_path} 文件中的数据全是 0！")
        
        plt.figure(figsize=(15, 6))
        
        # data.T 是 (126, 30)，我们把126个特征随30帧的变化画出来
        plt.plot(data.T) 
        
        plt.title(f"特征可视化: {title} ({file_path})", fontsize=16)
        plt.xlabel("帧 (时间)", fontsize=12)
        plt.ylabel("归一化的特征值", fontsize=12)
        plt.show()
        
    except Exception as e:
        print(f"加载文件 {file_path} 出错: {e}")

# --- 你需要修改这里 ---

# 1. 比较 "good" 和 "thank_you" (模型把 good 认成了 thank_you)
file_to_check_A = r"data/dataset/train/good/01/good_1_features.npy"
file_to_check_B = r"data/dataset/train/thank_you/01/thank_you_1_features.npy"

# 2. 比较 "morning" 和 "you" (模型把 morning 认成了 you)
file_to_check_C = r"data/dataset/train/morning/01/morning_1_features.npy"
file_to_check_D = r"data/dataset/train/you/01/you_1_features.npy"

# --- 运行诊断 ---
print("--- 正在诊断 'good' vs 'thank_you' ---")
visualize_features(file_to_check_A, "Good (A)")
visualize_features(file_to_check_B, "Thank You (B)")

print("\n--- 正在诊断 'morning' vs 'you' ---")
visualize_features(file_to_check_C, "Morning (C)")
visualize_features(file_to_check_D, "You (D)")
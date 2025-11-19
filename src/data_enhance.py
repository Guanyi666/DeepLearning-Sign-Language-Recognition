# ==================== final_complete_data_augment.py ====================
import os
import numpy as np
from scipy.interpolate import interp1d
import random
import shutil
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# --------------------- 全局配置 ---------------------
PLOT_KEYPOINT_ID = 8  # 食指指尖（第8号点），旋转效果最明显！

# --------------------- 安全的 seed 生成函数 ---------------------
def make_valid_seed(obj):
    """将任意对象转为合法的 NumPy seed (0 ~ 2**32-1)"""
    return hash(obj) & 0xFFFFFFFF

# --------------------- 数据增强核心函数 ---------------------
def augment_single_sequence(seq, seed=None):
    """
    seq: (30, 126)  两只手的关键点序列
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    s = seq.copy().astype(np.float32)  # (30, 126)

    # ------------------ 1. 时间拉伸 / 压缩 ------------------
    time_scale = np.random.uniform(0.88, 1.12)
    if abs(time_scale - 1.0) > 1e-3:
        old_t = np.linspace(0, 1, 30)
        new_t = np.clip(old_t * time_scale, 0, 1)
        f = interp1d(old_t, s, kind='linear', axis=0, fill_value="extrapolate")
        s = f(new_t)

    # ------------------ 2. 整体平移 ------------------
    translation = np.random.uniform(-0.1, 0.1, 3)
    s = s.reshape(30, 42, 3)   # 两只手共 42 点

    for h in range(2):
        hand_kp = s[:, h*21:(h+1)*21]  # (30,21,3)
        valid = np.any(hand_kp != 0, axis=2)

        for t in range(30):
            if np.any(valid[t]):
                hand_kp[t, valid[t]] += translation

    # ------------------ 3. 缩放 ------------------
    scale = np.random.uniform(0.88, 1.18)
    s *= scale

    # ------------------ 4. 手掌局部旋转 ------------------
    angle = np.random.uniform(-18, 18) * np.pi / 180
    if abs(angle) > 1e-3:
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ], dtype=np.float32)

        for h in range(2):
            hand = s[:, h*21:(h+1)*21]  # (30,21,3)

            for t in range(30):
                frame = hand[t]  # (21,3)
                valid = np.any(frame != 0, axis=1)

                if not np.any(valid):
                    continue

                # ---- 4.1 计算手掌中心（21 点平均） ----
                palm_center = np.mean(frame[valid, :2], axis=0, keepdims=True)  # (1,2)

                # ---- 4.2 局部坐标系旋转 ----
                xy = frame[valid, :2] - palm_center       # (N,2)
                xy_rot = xy @ R.T                         # (N,2)
                frame[valid, :2] = xy_rot + palm_center   # (N,2)

                hand[t] = frame

    # ------------------ 5. 加噪声 ------------------
    s += np.random.normal(0, 0.0035, s.shape)

    # reshape 回 30×126
    s = s.reshape(30, 126)
    return np.clip(s, -1.2, 2.0)


# --------------------- 批量增强 + 保留原始 + 完整统计报表 ---------------------
def batch_augment_keep_original(root_dir="../data/dataset/train",
                                output_dir="../data/dataset/train_aug",
                                aug_per_sample=8):
    os.makedirs(output_dir, exist_ok=True)

    before_count = {}   # 每个类别原始样本数
    after_count = {}    # 每个类别总样本数（含原始）
    total_before = 0
    total_after = 0

    for label in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue

        out_label_path = os.path.join(output_dir, label)
        os.makedirs(out_label_path, exist_ok=True)

        sample_cnt = 0
        for person_folder in os.listdir(label_path):
            person_path = os.path.join(label_path, person_folder)
            out_person_path = os.path.join(out_label_path, person_folder)
            os.makedirs(out_person_path, exist_ok=True)

            for file in os.listdir(person_path):
                if not file.endswith('.npy'):
                    continue

                src = os.path.join(person_path, file)
                dst = os.path.join(out_person_path, file)
                shutil.copy2(src, dst)  # 保留原始样本

                seq = np.load(src)
                sample_cnt += 1

                base_name = os.path.splitext(file)[0]
                for i in range(1, aug_per_sample + 1):
                    seed = make_valid_seed(f"{label}_{person_folder}_{base_name}_{i}")
                    aug_seq = augment_single_sequence(seq, seed=seed)
                    np.save(os.path.join(out_person_path, f"{base_name}_aug{i}.npy"), aug_seq)

        # 统计本类别
        before_count[label] = sample_cnt
        after_count[label] = sample_cnt * (1 + aug_per_sample)
        total_before += sample_cnt
        total_after += sample_cnt * (1 + aug_per_sample)

        print(f"✓ {label:12s} 原始 {sample_cnt:4d} → 增强后 {after_count[label]:5d} （含原始）")

    # --------------------- 最终统计报表 ---------------------
    print("\n" + "="*70)
    print("                  数据增强完成！已保留所有原始样本")
    print("="*70)
    print(f"原始路径       → {os.path.abspath(root_dir)}")
    print(f"增强输出路径   → {os.path.abspath(output_dir)}")
    print(f"每个样本生成增强数量：{aug_per_sample}")
    print("-"*70)
    print(f"{'类别':12s} {'原始数量':>8s} {'增强数量':>10s} {'总数量（含原始）':>15s}")
    print("-"*70)
    for label in sorted(before_count):
        print(f"{label:12s} {before_count[label]:8d} + {before_count[label]*aug_per_sample:8d} = {after_count[label]:12d}")
    print("-"*70)
    print(f"{'总计':12s} {total_before:8d} + {total_before*aug_per_sample:8d} = {total_after:12d}")
    print("="*70)
    print("   train_aug 文件夹现在可以直接用于训练啦！原始数据100%保留～")


# --------------------- 可视化函数（6宫格） ---------------------
def plot_augmentation_comparison(sample_path,
                                 save_path="../data/手语数据增强效果对比图_食指指尖X.png",
                                 num_aug=20):
    seq = np.load(sample_path)
    original_x = seq[:, PLOT_KEYPOINT_ID]

    methods = {
        "原始样本": [], "时间拉伸/压缩\n(Time Warping)": [],
        "高斯噪声抖动\n(Jittering)": [], "时间平移\n(Time Shift)": [],
        "手掌旋转\n(Rotation)": [], "组合增强\n(Combination)": []
    }

    for i in range(num_aug):
        np.random.seed(i + 1000)

        # 时间拉伸
        t = seq.copy().astype(np.float32)
        ts = np.random.uniform(0.85, 1.15)
        if abs(ts-1)>1e-3:
            f = interp1d(np.linspace(0,1,30), t, kind='cubic', axis=0, fill_value="extrapolate")
            t = f(np.clip(np.linspace(0,1,30)*ts, 0, 1))
        methods["时间拉伸/压缩\n(Time Warping)"].append(t[:, PLOT_KEYPOINT_ID])

        # 噪声
        methods["高斯噪声抖动\n(Jittering)"].append((seq + np.random.normal(0, 0.0035, seq.shape))[:, PLOT_KEYPOINT_ID])

        # 时间平移
        methods["时间平移\n(Time Shift)"].append(np.roll(seq, np.random.randint(-6,7), axis=0)[:, PLOT_KEYPOINT_ID])

        # 旋转 + 组合
        methods["手掌旋转\n(Rotation)"].append(augment_single_sequence(seq.copy(), seed=i+2000)[:, PLOT_KEYPOINT_ID])
        methods["组合增强\n(Combination)"].append(augment_single_sequence(seq.copy(), seed=i)[:, PLOT_KEYPOINT_ID])

    plt.figure(figsize=(18, 10))
    for idx, (title, trajs) in enumerate(methods.items()):
        plt.subplot(2, 3, idx+1)
        for traj in trajs:
            plt.plot(traj, color='lightblue' if idx==0 else plt.cm.tab10(idx), alpha=0.4, linewidth=1)
        plt.plot(original_x, 'k-', linewidth=3.5)
        plt.title(title, fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3)

    plt.suptitle('手语关键点数据增强效果对比（右手食指指尖 X 坐标轨迹）', fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n对比图已保存：{save_path}")


# ==================== 一键运行 ====================
if __name__ == "__main__":
    # # 1. 批量增强
    batch_augment_keep_original(
        root_dir="../data/dataset/train",
        output_dir="../data/dataset/train_aug",
        aug_per_sample=8   # 可改为 10、12 等
    )

    # 2. 生成对比图
    example = "../data/dataset/train/he/01/he_1_features.npy"
    plot_augmentation_comparison(
        sample_path=example,
        save_path="../data/手语数据增强_最终对比图.png",
        num_aug=22
    )
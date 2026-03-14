# DeepLearning-Sign-Language-Recognition 🤟

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://tensorflow.google.cn/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-green.svg)](https://google.github.io/mediapipe/)

本项目是一款基于深度学习的**双流 CNN-LSTM 融合网络手语识别系统**。作为深度学习课程的大作业，它实现了从原始视频采集、特征提取、模型训练到实时手语翻译的完整闭环解决方案。

---

## 📖 项目简介

本系统旨在解决手语交流中的障碍，通过计算机视觉技术将动态手语动作实时转化为中文句子。项目不仅关注单个手势的识别，更引入了时序建模与自然语言翻译模块，提升了系统的实用性。

---

## 🎯 主要功能

1.  **实时手语识别系统** (核心展示)
    * **多线程加速**：开启摄像头后，采用多线程架构确保视频采集与模型推理同步进行，无卡顿。
    * **语义翻译**：手部离开或静止约 2 秒，系统自动将识别出的单词序列翻译成通顺的中文句子。
    * **可视化反馈**：在摄像头画面中实时绘制中文识别结果。

2.  **自动化模型训练**
    * 支持读取预处理的 `.npy` 特征文件，一键开启训练并自动生成 Loss/Accuracy 曲线图。

3.  **高效数据预处理**
    * 利用 **MediaPipe** 提取 21 个手部关键点。
    * **能量分割算法**：根据运动能量自动定位手语动作段，无需手动标注起止帧。

---

## 🛠️ 技术路线

### 核心架构：双流 CNN-LSTM 融合网络
项目采用了创新的双流网络设计，以平衡空间特征与时间序列信息：
* **CNN (卷积神经网络)**：负责提取每一帧手势的空间姿态特征。
* **LSTM (长短期记忆网络)**：负责捕捉手语动作在时间跨度上的演变规律。
* **融合层**：高效结合空时特征，显著提升复杂动态手语的识别准确率。

### 技术栈清单
* **深度学习框架**：TensorFlow 2.20.0, Keras 3.10.0
* **计算机视觉**：MediaPipe 0.10.11, OpenCV 4.11.0
* **数据处理**：NumPy 1.26.4, Scikit-learn
* **UI/辅助**：Pillow (中文显示), Matplotlib (绘图)

---

## 💡 项目创新点

* **双流融合机制**：相比单一模型，更精准地描述了手语这种高度依赖“姿态+路径”的语言特征。
* **能量分割技术**：实现了数据清洗的半自动化，大幅降低了大规模手语数据集制作的门槛。
* **端到端闭环**：不仅仅是算法研究，更是一个可以直接运行的实用软件系统。
* **中文友好性**：解决了 OpenCV 无法原生显示中文的问题，提供了直观的中文交互界面。

---

## 📁 项目结构

```text
DeepLearning-Sign-Language-Recognition/
├── src/
│   ├── Camera/              # 实时识别模块 (核心程序)
│   │   └── realtime_detect_final.py
│   ├── Conv1D/              # 模型训练模块
│   │   └── run.py
│   └── process_video/       # 数据预处理模块
│       └── process_repeat_action_video.py
├── data/                    # 存放原始视频、.npy 特征及模型文件
├── requirements.txt         # 依赖项列表
└── main.py                  # 项目入口

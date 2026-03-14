# DeepLearning-Sign-Language-Recognition
本项目是一款基于深度学习的双流 CNN-LSTM 融合网络手语识别系统。通过结合空间特征提取（CNN）与时间序列建模（LSTM），实现了从原始视频采集到实时手语翻译的完整闭环解决方案。

🌟 项目亮点
双流融合架构：创新性地设计了双流网络，同步捕捉手部空间姿态与时序运动特征。

端到端工作流：涵盖数据预处理（能量分割）、模型训练、实时推理及自然语言翻译。

自动化动作切割：采用能量分割算法自动定位手语起止点，告别繁琐的人工标注。

高性能实时性：多线程异步处理摄像头流与模型预测，支持中文丝滑显示。

🛠️ 技术路线
本系统通过以下流程实现手语到文字的转换：

特征提取：利用 MediaPipe 提取高维度的手部关键点坐标，去除背景噪声。

动作分割：通过能量分割算法识别手部运动剧烈程度，自动截取有效动作片段。

核心模型：

CNN 层：提取每一帧手势的空间拓扑特征。

LSTM 层：学习动作在时间维度上的演变规律。

后处理：将识别出的单词序列通过翻译逻辑转换为连贯的中文句子。

📋 功能模块
1. 实时手语识别 (/src/Camera)
核心展示模块。开启摄像头后，系统会自动捕获手部动作，并在手部静止或离开后将识别结果拼凑成完整的中文句子。

运行：python src/Camera/realtime_detect_final.py

2. 模型训练 (/src/Conv1D)
读取预处理后的 .npy 特征文件，构建并训练 CNN-LSTM 模型。

运行：python src/Conv1D/run.py

输出：.h5 模型文件及训练过程可视化曲线（Loss/Accuracy）。

3. 数据预处理 (/src/process_video)
将原始录制的视频文件转化为模型可理解的特征向量。

运行：python src/process_video/process_repeat_action_video.py

📂 项目结构
Plaintext
DeepLearning-Sign-Language-Recognition/
├── data/                    # 原始视频与处理后的 .npy 特征数据
├── src/                     # 源代码目录
│   ├── Camera/              # 实时识别与 GUI 显示模块
│   ├── Conv1D/              # 模型定义与训练脚本
│   └── process_video/       # 视频读取与特征工程脚本
├── main.py                  # 项目统一入口（可选）
├── requirements.txt         # 依赖库清单
└── 深度学习大作业模版/        # LaTeX 论文报告模板 (NWPU)
🚀 快速开始
环境配置
建议使用 Python 3.8+ 环境，安装以下依赖：

Bash
pip install tensorflow==2.20.0 keras==3.10.0 mediapipe==0.10.11 opencv-python numpy==1.26.4 Pillow
快速运行
克隆项目：

Bash
git clone https://github.com/YourUsername/DeepLearning-Sign-Language-Recognition.git
cd DeepLearning-Sign-Language-Recognition
启动实时识别：

Bash
python src/Camera/realtime_detect_final.py
注：确保电脑已连接摄像头。按 Q 键退出。

💡 创新点详述
多线程优化：解决了 Python 全局解释器锁（GIL）对实时视频流造成的卡顿问题。

语义翻译层：不同于传统的单字识别，系统具备简单的上下文联想功能，将孤立的单词转化为可读性强的中文。

抗干扰能力：基于关键点的识别相比直接卷积像素，对环境光照和背景复杂度具有更强的鲁棒性。

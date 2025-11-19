import os
import numpy as np
import tensorflow as tf


class NLGProcessor:
    def __init__(self):
        # 1. 定义 英文(模型输出) -> 中文(NLG输入) 的映射字典
        self.en_to_cn_map = {
            "good": "好",
            "morning": "早上",
            "thank_you": "谢谢",
            "you": "你",
            "he": "他",
            "very": "很",
            # 如果有更多词，在这里添加
        }

        # 2. 尝试加载模型 (如果不存文件则跳过，只用规则)
        self.use_model = False
        try:
            # 假设模型在上一级目录的 sign_language_models 文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "..", "..", "src", "sign_language_models")

            model_path = os.path.join(model_dir, "intent_classifier_nlg.h5")

            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("✅ NLG 文本生成模型加载成功")
                self.use_model = True
            else:
                print("⚠️ 未找到 NLG 模型文件，将使用纯规则模式")
        except Exception as e:
            print(f"⚠️ NLG 模型加载失败: {e} (将使用纯规则模式)")

    def process(self, action_list):
        """
        核心入口：输入英文标签列表 -> 输出自然语言句子
        输入: ['morning', 'good']
        输出: "早上好"
        """
        if not action_list:
            return ""

        # 1. 将英文标签转换为中文单词
        # ['morning', 'good'] -> ['早上', '好']
        chinese_words = [self.en_to_cn_map.get(w, w) for w in action_list]
        print(f"   (NLG输入转化: {action_list} -> {chinese_words})")

        # 2. 调用翻译逻辑
        return self._translate_logic(chinese_words)

    def _translate_logic(self, words):
        """
        这里是你提供的 main.py 里的核心逻辑
        """

        # 辅助函数：去重与排序
        def preprocess(ws):
            # 定义优先级顺序
            order = ["早上", "谢谢", "我", "你", "他", "她", "很", "好"]
            seen = set()
            result = []
            # 按优先级提取
            for w in order:
                if w in ws and w not in seen:
                    result.append(w)
                    seen.add(w)
            # 如果都不在优先级里，保持原样
            return result if result else ws

        # 预处理
        words = preprocess(words)
        s = set(words)

        # --- 规则匹配系统 (Rule Based) ---

        # 场景 1: 问候
        if "早上" in s:
            if "谢谢" in s:
                return "早上好，谢谢你" if "你" in s else "早上好，谢谢"
            return "早上好"

        # 场景 2: 感谢
        if "谢谢" in s:
            if "你" in s: return "谢谢你"
            if "他" in s: return "谢谢他"
            return "谢谢"

        # 场景 3: 夸奖/状态
        if "很" in s and "好" in s:
            if "你" in s: return "你很好"
            if "他" in s: return "他很好"
            return "很好"

        if "好" in s:
            if "你" in s: return "你好"
            if "他" in s: return "他好"
            return "好"

        # 默认兜底
        return "".join(words)


# 用于单独测试
if __name__ == "__main__":
    nlg = NLGProcessor()
    print(nlg.process(["morning", "good"]))  # 应该输出：早上好
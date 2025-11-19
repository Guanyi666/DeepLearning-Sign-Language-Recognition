import time


class SignTranslator:
    def __init__(self):
        # 1. 定义词汇映射规则 (Gloss -> Natural Language)
        # 这里的键 (Key) 是识别到的单词序列（用空格连接）
        # 值 (Value) 是翻译后的自然语言
        self.rules = {
            "you good": "你好",
            "you very good": "你很棒",
            "he good": "他很好",
            "good morning": "早上好",
            "morning good": "早上好",
            "morning": "早安",
            "thank_you": "谢谢",
            "you thank_you": "谢谢你",
            "thank_you you": "谢谢你",
            "he thank_you": "他表示感谢",
            "thank_you he": "他表示感谢",
            "he very good": "他非常优秀",
            "you": "你",
            "he": "他",
            "good": "好",
            "very": "非常",
            "good you": "你好",
            "you good you": "你好",
            "good good you": "你好",
            "you you good": "你好"


        }

        self.word_buffer = []  # 存储当前的单词序列
        self.last_update_time = 0
        self.TIMEOUT = 5  # 如果 2.5 秒没有新词，就自动提交翻译

    def add_word(self, word):
        """
        接收实时识别到的词
        """
        # 简单的去重逻辑：如果这个词和上一个词一样，就不加了
        # (防止手没放下，连续识别出 you you you)
        if not self.word_buffer or self.word_buffer[-1] != word:
            self.word_buffer.append(word)
            self.last_update_time = time.time()
            return True  # 返回 True 表示有新词加入
        return False

    def translate(self):
        """
        将当前的单词缓冲区翻译成句子
        """
        if not self.word_buffer:
            return ""

        # 1. 拼接成字符串 key: "you good"
        gloss_sequence = " ".join(self.word_buffer)

        # 2. 查找规则
        result = ""
        if gloss_sequence in self.rules:
            result = self.rules[gloss_sequence]
        else:
            # 如果没找到匹配的规则，就直接把单词拼起来 (兜底策略)
            # 比如识别了 "he morning"，规则里没有，就输出 "他 Morning"
            result = gloss_sequence

            # 3. 翻译完成后清空缓冲区
        self.word_buffer = []
        return result

    def check_auto_submit(self):
        """
        [修改] 检查是否超时。
        如果超时，返回【当前积累的所有动作列表】，并清空缓冲区。
        """
        # 判断是否非空且超时
        if self.word_buffer and (time.time() - self.last_update_time > self.TIMEOUT):
            # 1. 拷贝一份列表 (因为马上要清空 self.word_buffer)
            final_sequence = self.word_buffer.copy()

            # 2. 清空缓冲区，准备接收下一句话
            self.word_buffer = []

            # 3. 返回原始列表，例如 ['you', 'very', 'good']
            return final_sequence

        return None

    def rules_translate(self, sequence_list):
        """
        [新增] 纯粹的翻译功能：输入列表 -> 输出中文
        """
        gloss_sequence = " ".join(sequence_list)
        if gloss_sequence in self.rules:
            return self.rules[gloss_sequence]
        return gloss_sequence  # 没查到就返回原文
import jieba
from sklearn.model_selection import train_test_split


class stopword(object):
    def __init__(self, file):
        fd = open(file, encoding='utf-8')
        self.data = []
        for word in fd:
            self.data.append(word.strip())


class Data(object):
    def __init__(self, train_file, stop_file = None):
        # 去停用词
        def remove_stopwords(stop_list, data_list):
            result = []
            # 这里面的leave_word不能作为最终的特征，因为leave_word当中每个词只出现一次
            # 当多个非停用词出现在一句话中的时候，信息就会损失
            leave_word = set(data_list) - set(stop_list)
            for word in data_list:
                if word in leave_word:
                    result.append(word)
            return result

        self.data = []
        self.label = []

        file = open(train_file, "r", encoding='utf-8')
        if stop_file is not None:
            self.stopwords = set(stopword(stop_file).data)
        else:
            self.stopwords = set()

        for line in file:
            split_line = line.split('\t')

            if len(split_line) != 2:
                print(line)
                continue

            label, content = split_line
            content = remove_stopwords(self.stopwords, list(jieba.cut(content)))
            content = ' '.join(content)
            self.data.append(content)
            self.label.append(label)


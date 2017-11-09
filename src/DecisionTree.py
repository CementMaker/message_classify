from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from PreProcess import *

from sklearn.tree import export_graphviz

def DecisionTreeClassifyTfidf(normal, spam, stop):
    data = Data(normal, spam, stop)
    # 统计每个词的TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 max_features=5000,
                                 min_df=2,
                                 use_idf=True,
                                 lowercase=False,
                                 decode_error='ignore',
                                 analyzer=str.split).fit(data.data)
    train_x, test_x, train_y, test_y = train_test_split(data.data, data.label, test_size=0.1)
    train_x = vectorizer.transform(train_x)
    test_x = vectorizer.transform(test_x)

    DecisionTree = DecisionTreeClassifier(criterion="entropy",
                                          splitter="best",
                                          max_depth=None,
                                          min_samples_split=2,
                                          min_samples_leaf=2)
    # 训练加上测评
    DecisionTree.fit(train_x, train_y)
    print(DecisionTree.score(test_x, test_y))

    # 特征提取，提取词汇
    words = vectorizer.get_feature_names()
    feature_importance = DecisionTree.feature_importances_
    word_importances_dict = dict(zip(words, feature_importance))

    number = 0
    for word, importance in sorted(word_importances_dict.items(), key=lambda val:val[1], reverse=True):
        print(word, importance)
        number += 1
        if number == 200:
            break


def DecisionTreeClassifyTf(normal, spam, stop):
    data = Data(normal, spam, stop)

    # 统计每个词的频率
    vectorizer = CountVectorizer(max_df=0.6,
                                 max_features=5000,
                                 min_df=2,
                                 decode_error='ignore',
                                 analyzer=str.split).fit(data.data)
    train_x, test_x, train_y, test_y = train_test_split(data.data, data.label, test_size=0.1)
    train_x = vectorizer.transform(train_x)
    test_x = vectorizer.transform(test_x)

    DecisionTree = DecisionTreeClassifier(criterion="entropy",
                                          splitter="best",
                                          max_depth=None,
                                          min_samples_split=2,
                                          min_samples_leaf=2)
    # 训练加上测评
    DecisionTree.fit(train_x, train_y)
    print(DecisionTree.score(test_x, test_y))

    # 特征提取，提取词汇
    words = vectorizer.get_feature_names()
    feature_importance = DecisionTree.feature_importances_
    word_importances_dict = dict(zip(words, feature_importance))

    number = 0
    for word, importance in sorted(word_importances_dict.items(), key=lambda val:val[1], reverse=True):
        print(word, importance)
        number += 1
        if number == 200:
            break

    # 画出决策树，tree.dot还不是图片，需要用命令dot
    from sklearn.tree import export_graphviz
    export_graphviz(DecisionTree, open("tree.dot", "w"))
    # dot -Tpng tree.dot -o tree.png


# DecisionTreeClassifyTfidf('../data/normal.txt', '../data/spam.txt', "../data/stopword.txt")
DecisionTreeClassifyTf('../data/normal.txt', '../data/spam.txt', "../data/stopword.txt")

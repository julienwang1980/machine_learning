# coding:utf-8

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def dict_demo():
    """
    字典特征提取
    :return: None
    """
    # 1.获取数据
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]

    # 2.字典特征提取
    # 2.1 实例化
    # transfer = DictVectorizer()
    transfer = DictVectorizer(sparse=False)

    # 2.2 转换
    new_data = transfer.fit_transform(data)
    print(new_data)

    # 2.3 获取具体属性名
    names = transfer.get_feature_names_out()
    print("属性名字是:\n", names)


def english_count_demo():
    """
    文本特镇提取-英文
    :return: None
    """
    # 获取数据
    data = ["life is is short,i like python",
            "life is too long,i dislike python"]

    # 文本特征转换
    # transfer = CountVectorizer(sparse=True) # no parameter of "sparse"
    transfer = CountVectorizer(stop_words=["dislike"])
    new_data = transfer.fit_transform(data)

    # 查看特征名字
    names = transfer.get_feature_names_out()

    print("特征名字是:\n", names)
    print(new_data.toarray())
    print(new_data)



if __name__ == '__main__':
    # dict_demo()
    english_count_demo()

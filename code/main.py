import re
import time
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from scipy.spatial.distance import pdist
pd.set_option('display.max_columns',None)


def clean_data():
    '''
    合并需要的文本并保存，读取保存数据
    :return: 索引，相似度计算的文本
    '''
    # 读取数据并选择列
    df = pd.read_csv('../data/cat_kind.csv')
    df = df[['品种','整体','颜色','头部','眼睛','耳朵','鼻子','尾巴',
             '胸部','颈部','前驱','后驱','基本信息','FCI标准','性格特点','生活习性',
             '优点/缺点','喂养方法','鉴别挑选']]

    # nan 填充，所有描述类文本合为一列
    df = df.fillna('')
    df['描述'] = df['整体'] + df['颜色'] + df['头部'] + df['眼睛'] + df['耳朵'] \
        + df['鼻子'] + df['尾巴'] + df['胸部'] + df['颈部'] + df['前驱'] \
        + df['后驱'] + df['基本信息'] + df['FCI标准'] + df['性格特点'] + df['生活习性'] \
        + df['优点/缺点'] + df['喂养方法'] + df['鉴别挑选']
    df = df[['品种','描述']].iloc[:-1,:]

    # 保存
    df.to_csv('../output/data.csv',index=False)

    # 读取数据
    df = pd.read_csv('../output/data.csv')

    # 行列索引，文本
    index = list(df['品种'])
    text = list(df['描述'])

    return index,text


def difflib(text):
    '''
    内置库 difflib 相似度计算，每一个文本两两计算
    :param text: 文本列表
    :return:
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            sequenceMatcher = SequenceMatcher()
            sequenceMatcher.set_seqs(t,text[i])
            score = sequenceMatcher.ratio() # 计算相似度
            s.append(round(score,2))
        similarity.append(s)

    return similarity


def fuzzywuzzy(text):
    '''
    fuzzywuzzy 库计算相似度，文本两两相似度
    :param text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            score = fuzz.ratio(t, text[i]) # 计算相似度
            s.append(score)
        similarity.append(s)

    return similarity


def cos(text):
    '''
    余弦距离 相似度计算
    :param text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i]) # 获得两个文本词向量
            score = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) # 计算相似度
            s.append(round(score,2))
        similarity.append(s)

    return similarity


def oushi(text):
    '''
    欧式距离 相似度计算
    :param text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i])  # 获得两个文本词向量
            v1,v2 = np.mat(v1),np.mat(v2)
            score = float(np.sqrt(np.sum(np.square(v1-v2))))  # 计算相似度
            s.append(round(score, 2))
        similarity.append(s)

    return similarity


def mah(text):
    '''
    曼哈顿距离
    相似度计算
    :param
    text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i])  # 获得两个文本词向量
            v1,v2 = np.mat(v1),np.mat(v2)
            score = float(np.sum(np.abs(v1-v2)))  # 计算相似度
            s.append(round(score, 2))
        similarity.append(s)

    return similarity


def cheb(text):
    '''
    切比雪夫距离 相似度计算
    :param
    text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i])  # 获得两个文本词向量
            v1, v2 = np.mat(v1), np.mat(v2)
            score = float(np.max(np.abs(v1-v2)))  # 计算相似度
            s.append(round(score, 2))
        similarity.append(s)

    return similarity


def yac(text):
    '''
    杰尔德距离 相似度计算
    :param
    text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i])  # 获得两个文本词向量
            Vec = np.vstack([v1,v2])
            score = pdist(Vec,'jaccard')  # 计算相似度
            s.append(round(score[0], 2))
        similarity.append(s)

    return similarity


def han(text):
    '''
    汉明距离 相似度计算
    :param
    text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i])  # 获得两个文本词向量
            Vec = np.vstack([v1, v2])
            score = pdist(Vec, 'hamming')  # 计算相似度
            score = score[0]*len(v1)
            s.append(round(score, 2))
        similarity.append(s)

    return similarity


def corrcoef(text):
    '''
    皮尔逊相关系数 相似度计算
    :param
    text: 文本列表
    :return: 相似度
    '''
    similarity = []
    for t in text:
        s = []
        for i in range(len(text)):
            v1, v2 = get_word_vector(t, text[i])  # 获得两个文本词向量
            Vec = np.vstack([v1, v2])
            score = np.corrcoef(Vec)[0][1]  # 计算相似度
            s.append(round(score, 2))
        similarity.append(s)

    return similarity


def get_word_vector(s1, s2):
    """
    :param s1: 字符串1
    :param s2: 字符串2
    :return: 返回字符串切分后的向量
    """
    # 字符串中文按字分，英文按单词，数字按空格
    regEx = re.compile('[\\W]*')
    res = re.compile(r"([\u4e00-\u9fa5])")
    p1 = regEx.split(s1.lower())
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
    # print(str1_list)
    p2 = regEx.split(s2.lower())
    str2_list = []
    for str in p2:
        if res.split(str) == None:
            str2_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str2_list.append(ch)
    # print(str2_list)
    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    list_word2 = [w for w in str2_list if len(w.strip()) > 0]  # 去掉为空的字符
    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))
    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    return word_vector1, word_vector2


def save(index,similarity,name):
    '''
    相似度结果转为矩阵保存
    :param similarity: 两两相似度
    :param name: 相似度计算方法
    :return:
    '''
    df = pd.DataFrame(index=index)
    for s in enumerate(similarity):
        df[index[s[0]]] = s[1]
    df.to_csv('../output/'+name+'.csv')


def main():
    '''
    主函数
    :return:
    '''
    # 索引保存为二维矩阵用，相似度计算文本
    index,text = clean_data()

    # difflib 计算相似度
    start = time.time()
    similarity = difflib(text)
    save(index,similarity,'difflib')
    print('difflib 相似度计算完毕~ %s 秒' % (round(time.time()-start,2)))

    # fuzzywuzzy 计算相似度
    start = time.time()
    similarity = fuzzywuzzy(text)
    save(index,similarity,'fuzzywuzzy')
    print('fuzzywuzzy 相似度计算完毕~ %s 秒' % (round(time.time()-start,2)))

    # 余弦相似度 计算相似度
    start = time.time()
    similarity = cos(text)
    save(index, similarity, '余弦')
    print('余弦 相似度计算完毕~ %s 秒' % (round(time.time()-start,2)))

    # 欧氏距离
    start = time.time()
    similarity = oushi(text)
    save(index, similarity, '欧氏距离')
    print('欧氏距离 相似度计算完毕~ %s 秒' % (round(time.time() - start, 2)))

    # 曼哈顿距离
    start = time.time()
    similarity = mah(text)
    save(index, similarity, '曼哈顿距离')
    print('曼哈顿距离 相似度计算完毕~ %s 秒' % (round(time.time() - start, 2)))

    # 切比雪夫距离
    start = time.time()
    similarity = cheb(text)
    save(index, similarity, '切比雪夫距离')
    print('切比雪夫距离 相似度计算完毕~ %s 秒' % (round(time.time() - start, 2)))

    # 杰尔德距离
    start = time.time()
    similarity = yac(text)
    save(index, similarity, '杰尔德距离')
    print('杰尔德距离 相似度计算完毕~ %s 秒' % (round(time.time() - start, 2)))

    # 汉明距离
    start = time.time()
    similarity = han(text)
    save(index, similarity, '汉明距离')
    print('汉明距离 相似度计算完毕~ %s 秒' % (round(time.time() - start, 2)))

    # 皮尔逊相关系数
    start = time.time()
    similarity = corrcoef(text)
    save(index, similarity, '皮尔逊相关系数')
    print('皮尔逊相关系数 相似度计算完毕~ %s 秒' % (round(time.time() - start, 2)))


if __name__ == '__main__':
    main()

# -*- coding:utf-8 -*-
import collections
import random
import jieba
import numpy as np
import re
import tensorflow as tf  # tensorflow_version 2.x


class Myword2vec:

    def __init__(self, skip_num=2, skip_window=3,
                 batch_size=128, min_occurence=10,
                 embedding_size=100, max_vocabulary_size=100000,
                 negative_sampled=64):
        """
        本代码实现了word embedding中的skip-gram方法，CBOW（continue bag of word）方法只需修改
        gen_batch中的部分代码即可实现
        :param skip_num: skip-gram中需要使用的word数量
        :param skip_window: 假如skip_window=5, 则最中间的词为要预测的词，左右各两次为中间词的标签
        :param batch_size: 批数量
        :param min_occurence: 建立词典时需要每个词出现的最小次数，最小次数以下的词不记录到词典中
        :param embedding_size: 词向量维度 （建议为log2(vocabulary_size))
        :param max_vocabulary_size: 词典最大容量
        :param negative_sampled: 计算Nec loss时的采样数量(Noise contrastive estimation)
        """
        self.skip_num = skip_num
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.min_occurence = min_occurence
        self.embedding_size = embedding_size
        self.negative_sampled = negative_sampled
        self.max_vocabulary_size = max_vocabulary_size
        self.data_index = 0

    def split_words(self, text_path, stop_word_path, lang='chinese'):
        """
        分词，其中中文分词使用的Jieba分词工具
        :param text_path: 文档地址（词向量的词从文档中得出）
        :param stop_word_path: 停用词地址
        :param lang: 语言模式（'chinese','english'）
        :return: 词列表
        """
        word_list = []
        stop_sign = "[：\(\)《》…… ？“” ，。·；\- ☆ ■、！※\‘\’\da-zA-Z#$%&@~\\n...\n,.:;\'\"\\\[\]\{\}_]"
        stop_words = [line.strip() for line in open(stop_word_path, 'r').readlines()]
        with open(text_path, 'r', decode='utf-8') as f:
            line = f.readline()
            while line:
                if lang == 'chinese':
                    line = re.sub(stop_sign, '', line)
                    raw_words = [word for word in jieba.cut(line) if word not in stop_words]
                    word_list.extend(raw_words)
                    line = f.readline()
                else:
                    line = re.sub(stop_sign, ' ', line)
                    raw_words = line.lower().split(' ')
                    word_list.extend(raw_words)
                    line = f.readline()
        return word_list

    def build_vocabulary(self, word_list):
        """
        建立词典
        :param word_list: 词列表
        :return: 词id列表、词到id的字典、id到词的字典、词频统计、词典大小
        """
        word2id = {}
        count = [('UNK', 1)]
        count.extend(collections.Counter(word_list).most_common(self.max_vocabulary_size - 1))
        for i in range(len(count) - 1, -1, -1):
            if count[i][1] < self.min_occurence:
                count.pop(i)
            else:
                break

        vocabulary_size = len(count)

        for i, (word, _) in enumerate(count):
            word2id[word] = i
        unknownWordCount = 0
        id_list = []
        for word in word_list:
            index = word2id.get(word, 0)
            id_list.append(index)
            if index == 0:
                unknownWordCount += 1
        count[0][1] = unknownWordCount
        id2word = dict(zip(word2id.values(), word2id.keys()))
        return id_list, word2id, id2word, count, vocabulary_size

    def gen_batch(self, id_list):
        """
        创建批数据
        :param id_list: 词id列表
        :return: 批数据、批数据对应的标签
        """
        batch = np.ndarray(shape=(self.batch_size, ), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        assert self.batch_size % self.skip_num == 0
        assert self.skip_num <= 2 * self.skip_window
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(id_list):
            self.data_index = 0
        buffer.extend(id_list[self.data_index: span + self.data_index])
        self.data_index = (self.data_index + span) % len(id_list)
        for i in range(self.batch_size // self.skip_num):
            label_words = [index for index in range(span) if i != self.skip_window]
            label2use = random.sample(label_words, self.skip_num)
            for j, label in enumerate(label2use):
                batch[i * self.skip_num + j] = buffer[self.skip_window]
                labels[i * self.skip_num + j, 0] = buffer[label]
            buffer.append(id_list[self.data_index])
            self.data_index = (self.data_index + 1) % len(id_list)
        return batch, labels

    def get_embedding(self, embedding, x):
        """
        根据词id获取对应的词向量
        :param embedding: 词向量(Matrix)
        :param x: 词id
        :return: 词向量（Vector）
        """
        x_embedd = tf.nn.embedding_lookup(embedding, x)
        return x_embedd

    def nce_loss(self, x_embedd, y, nce_weight, nce_bias, vocabulary_size):
        """
        计算NCE损失，如果有GPU资源，可将tf.device中的'/cpu:0'换为GPU
        :param x_embedd: x词的词向量
        :param y: x词对应的标签
        :param nce_weight: nce权重
        :param nce_bias: nce偏置项
        :param vocabulary_size: 词典大小
        :return: 损失大小
        """
        with tf.device("/cpu:0"):
            y = tf.cast(y, tf.int64)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    inputs=x_embedd,
                    labels=y,
                    weights=nce_weight,
                    biases=nce_bias,
                    num_class=vocabulary_size,
                    num_sampled=self.negative_sampled,
                )
            )
        return loss

    def simlarity_eval(self, test_embedd, embedding):
        """
        计算测试用的词汇的语义相近的其他词汇
        :param test_embedd: 测试词汇的词向量(Vector)
        :param embedding: 词向量（Matrix)
        :return: cosine 相似度
        """
        with tf.device("/cpu:0"):
            test_embedd = tf.cast(test_embedd, tf.float32)
            test_norm = test_embedd / tf.sqrt(tf.reduce_sum(tf.square(test_embedd)))
            embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding)))
            cosine_sim = tf.matmul(test_norm, embedding_norm, transpose_b=True)
        return cosine_sim

    def optimization(self, x, y, embedding, nce_weight, nce_bias):
        """
        优化器，本代码中使用Adadelta
        :param x: 词向量（Vector）
        :param y: 词向量对应的标签
        :param embedding: 词向量
        :param nce_weight: nce权重
        :param nce_bias: nce偏置项
        """
        optimizer = tf.optimizers.Adadelta()
        with tf.device("/cpu:0"):
            with tf.GradientTape() as g:
                x_embedd = self.get_embedding(embedding, x)
                loss = self.nce_loss(x_embedd, y)
            gradient = g.gradient(loss, [embedding, nce_weight, nce_bias])
            optimizer.apply_gradients(zip(gradient, [embedding, nce_weight, nce_bias]))

    def run(self, text_path, stop_word_path, batch_size, embedding_dim, test_words=[], epochs=100000, verbos=True):
        """
        WordEmbedding训练入口
        :param text_path: 文档地址
        :param stop_word_path: 停用词地址
        :param batch_size: 批大小
        :param embedding_dim: 词向量维度
        :param test_words: 测试词汇列表
        :param epochs: 训练次数
        :param verbos: 训练进度输出（True、False）
        """
        display_step = epochs / 100
        eval_step = display_step * 2
        word_list = self.split_words(text_path, stop_word_path)
        id_list, word2id, id2word, count, vocabulary_size = self.build_vocabulary(word_list)
        embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_dim]))
        nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_dim]))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        for epoch in range(epochs):
            batch_x, batch_y = self.gen_batch(id_list, batch_size)
            self.optimization(batch_x, batch_y, embedding, nce_weights, nce_biases)
            if verbos is False: continue
            if epoch % display_step == 0 or epoch == 1:
                loss = self.nce_loss(self.get_embedding(embedding, batch_x), batch_y)
                print("step: {}, loss: {}".format(epoch, loss))
            if epoch % eval_step == 0 or epoch == 1:
                print("Test set simlarity evaluation")
                simlarity = self.simlarity_eval(self.get_embedding(embedding, test_words), embedding).numpy()
                for i in range(len(test_words)):
                    top_k = 5
                    nearest_word = -(simlarity[i, :]).argsort()[1:top_k + 1]
                    words = [id2word[i] for i in nearest_word]
                    string = ','.join(words)
                    final_str = '"%s" nearest neighbors: %s' % (test_words[i], string)
                    print(final_str)
        self.embedding = embedding


if __name__ == "__main__":
    text_path = ''
    stop_word_path = ''
    batch_size = 128
    test_words =[]
    model = Myword2vec()
    model.run(text_path=text_path, stop_word_path=stop_word_path, batch_size=batch_size, test_words=test_words)

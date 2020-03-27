# -*- coding:utf-8 -*-
###########################################################################################
# 本文基本模仿自https://github.com/NELSONZHAO/zhihu中的Seq2Seq Attention Keras 实现方法       #
# 经过一段时间的吴恩达、李宏毅老师的视频课程学习，决定动手实现一下代码，但是发现一到动手写代码的        #
# 时候就觉的无从下手，这可能也是只看理论不动手敲代码实践的结果，因此从网上找公开的资料从头实现一        #
# 遍。 本代码实现的英法翻译中的Attention中Encoder中的HiddenState值与Decoder中的HiddenSate       #
# 值的相似度计算采用了（1：点积 Dot; 2:cosine相似度；3：MLP网络（多层感知机））中的1方法。          #
# 关于Attention中相似度计算的三种方式，可移步https://www.cnblogs.com/guoyaohua/p/9429924.html  #
# 再次感谢🙏博主的代码, 博主知乎上的精彩分享地址：https://zhuanlan.zhihu.com/p/37290775         #
###########################################################################################

import os, re, jieba, tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras.backend as K
from datetime import datetime
from keras.optimizers import adam
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Concatenate, Reshape, Dot, RepeatVector, Activation
from keras.layers import Input, Dense, LSTM, Bidirectional, Embedding
from nltk.translate.bleu_score import sentence_bleu

# 训练资料可到本文开头的Github地址去clone
englishText_path = 'zhihu/mt_attention_birnn/data/small_vocab_en'
frenchText_path ='zhihu/mt_attention_birnn/data/small_vocab_fr'
# glove采用哈佛大学100维的词向量, 下载地址可在网上找到镜像去下载，原网址下载速度太慢！
glove_path = '/downloads/glove.6B/glove.6B.100d.txt'


def text2dict(path, source=True):
    """
    :param path: 文档地址
    :param source: 是否为源文档，bool值（True/False）
    :return:
    """
    # '<PAD>'代表补位标记, '<UNK>'代表未录入词典的未知词汇
    # '<BOS>'代表句子开头, '<EOS>'代表句子末尾
    sourcepattern = ['<PAD>', '<UNK>']
    targetpattern = ['<PAD>', '<UNK>', '<BOS>', 'EOS']
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = text.split('\n')
    sentencescount = np.sum(sentences)
    wordscount = [len(sentence.split(' ')) for sentence in sentences]
    totalwordscount = np.sum(wordscount)
    averagewordsnumpersent = np.average(wordscount)
    maxwordsnumpersent = np.max(wordscount)
    vocabulary = list(set(text.lower().split()))
    if source is True:
        word2id = {word: idx for idx, word in enumerate(sourcepattern + vocabulary)}
        id2word = {idx: word for word, idx in word2id.items()}
    else:
        word2id = {word: idx for idx, word in enumerate(targetpattern + vocabulary)}
        id2word = {idx: word for word, idx in word2id.items()}
    dictsize = len(word2id)
    print('---' * 15)
    print('The number of sentence in this text is {}'.format(sentencescount))
    print('The number of words in this text is {}'. format(totalwordscount))
    print('The average number of words in a sentence is {}'.format(averagewordsnumpersent))
    print('The max number of words in a sentence is {}'.format(maxwordsnumpersent))
    print('The dict size of this text is {}'.format(dictsize))
    print('---' * 15)
    return sentences, word2id, id2word, dictsize, maxwordsnumpersent


def sentence2id(sentence, word2id, maxlen=20, source=True):
    """
    :param sentence: 将句子字符编码为数字标识符
    :param word2id: key为word, value为id的字典
    :param maxlen: 句子的最大长度（当句子长度大于maxlen时，
                   截取maxlen之前的标识符，当句子长度小于maxlen时补'<PAD>'）
    :param source: 是否为源文档，bool值（True/False）
    :return:
    """
    idx = []
    unk_id = word2id.get('<UNK>')
    pad_id = word2id.get('<PAD>')
    eos_id = word2id.get('<EOS>')
    if source is True:
        for word in sentence.lower().split():
            idx.append(word2id.get(word, unk_id))
    else:
        for word in sentence.lower().split():
            idx.append(word2id.get(word, unk_id))
        idx.append(eos_id)
    if len(idx) > maxlen:
        idx = idx[:maxlen]
    else:
        idx = idx + [pad_id] * (maxlen - len(idx))
    return idx


def text2id(path, source=True):
    """
    :param path: 文档路径
    :param source: 是否为源文档，bool值（True/False）
    :return:
    """
    sentences, word2id, id2word, dictsize, maxlen = text2dict(path, source=source)
    textidx = []
    for sentence in tqdm.tqdm(sentences):
        idx = sentence2id(sentence, word2id, maxlen=maxlen, source=source)
        textidx.append(idx)
    textidx = np.array(textidx, dtype=np.float32)
    print('The text words have been translated into unique id successfully !')
    return textidx, word2id, id2word, dictsize, maxlen


def getGlove(path):
    """
    :param path: glove文档存储地址
    :return: key为word, value为词所对应的词向量
    """
    with open(path, 'r', encoding='utf-8') as f:
        word2vect = {}
        for line in f:
            line = line.strip().split()
            word = line[0]
            word2vect[word] = np.array(line[1:], dtype=np.float32)
    print('The glove text has been converted into word2vector dict !')
    return word2vect


def pretrainedembedding(path, word2id, maxlen):
    """
    :param path: glove文档存储地址
    :param word2id: key为word, value为id的字典
    :param maxlen: 输入句子的最大长度
    :return: keras embeddinglayer
    """
    word2vect = getGlove(path)
    # Keras API要求的 Embedding size 要比字典大一维
    vocabulary_size = len(word2id) + 1
    # 词向量维度（本文中使用的glove维度为100）
    embedding_dim = word2vect['the'].shape[0]
    embedding_matrix = np.zeros(shape=(vocabulary_size, embedding_dim), dtype=np.float32)
    for word, idx in word2id.items():
        vector = word2vect.get(word, np.zeros(embedding_dim))
        embedding_matrix[idx, :] = vector
    embeddinglayer = Embedding(vocabulary_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=maxlen,
                               trainable=False, name='PreTrainedEmbedding')
    print('Pre-trained embedding layer loaded completely !')
    return embeddinglayer


# 最好给keras每个layer起名字，后续查看每个layer输出时比较方便
def attention(h, s, sourceSentenceLen):
    """
    :param h: encoder隐层状态值（h为t时刻的值）
    :param s: decoder隐层状态值（s为t-1时刻的值）
    :param sourceSentenceLen: encoder输入长度
    :return: cotext值
    """
    s = RepeatVector(sourceSentenceLen, name='Repeat_s')(s)
    concat = Concatenate(axis=-1, name='Concateh&s')([h, s])
    densetanh = Dense(32, activation='tanh', name='DenseTanh32')(concat)
    denserelu = Dense(1, activation='relu', name='DenseRelu1')(densetanh)
    alphas = Activation(activation='softmax', name='attentionWeight')(denserelu)
    context = Dot(axes=1, name='ContextByDot')([alphas, h])
    return context


def encoderBiLSTM(units):
    """
    :param units: encoder LSTM unit个数， BidirectionalLTSM的unit个数为2倍的LSTM unit
    :return: encoder
    """
    biLSTM = Bidirectional(LSTM(units, return_sequences=True))
    return biLSTM


def decoderLSTM(units):
    """
    :param units: decoder LSTM unit个数
    :return: decoder
    """
    decoder = LSTM(units, return_state=True)
    return decoder


def seq2seqwithAttention(sourceSentenceLen, targetSentenceLen,
                         encoderUnitsnum, decoderUnitsnum,
                         sourceVocabsize, targetVocabsize,
                         embeddingLayer):
    """
    :param sourceSentenceLen: 源语句最大长度
    :param targetSentenceLen: 目标语句最大长度
    :param encoderUnitsnum: encoder中LSTM数量
    :param decoderUnitsnum: decoder中LSTM数量
    :param sourceVocabsize: 源语言字典大小（包括'<PAD>', '<UNK>'）
    :param targetVocabsize: 目标语言字典大小（包括'<PAD>', '<UNK>', '<BOS>', '<EOS>'）
    :param embeddingLayer: 预先训练好的Embedding层
    :return: model
    """
    x = Input(shape=(sourceSentenceLen, ), name='EncoderInput')
    x = embeddingLayer(x)
    s = Input(shape=(decoderUnitsnum, ), name='DecoderInitialSInput')
    c = Input(shape=(decoderUnitsnum, ), name='DecoderInitialCInput')
    out = Input(shape=(targetSentenceLen, ), name='DecoderInitialOutput')
    out = Reshape((1, targetSentenceLen), name='DecoderOutputReshaper')(out)

    encoder = encoderBiLSTM(encoderUnitsnum)
    decoder = decoderLSTM(decoderUnitsnum)
    outputer = Dense(targetVocabsize, activation='softmax', name='DecoderOutput')

    outputs = []
    h = encoder(x)
    for i in range(targetSentenceLen):
        context = attention(h, s, sourceSentenceLen)
        decoderInput = Concatenate(axis=-1)([context, Reshape((1, targetSentenceLen))[out]])
        s, _, c = decoder(decoderInput, initial_state=[s, c])
        out = outputer(s)
        outputs.append((out))

    model = Model(input=[x, s, c, out], output=[outputs], name='seq2seqWithAttention')
    model.compile(optimizer=adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model


def prediction(sourceSentence, sourceword2id, sourceSentenceLen, decoderUnitsnum, targetSentenceLen, model):
    """
    :param sourceSentence: 要翻译的源句子
    :param sourceword2id: 源语言字典
    :param sourceSentenceLen: 源句子长度（padding后的长度）
    :param decoderUnitsnum: decoder中LSTM数量
    :param targetSentenceLen: 目标语句最大长度
    :param model: 训练好的模型
    :return: 翻译好的目标语句
    """

    s = np.zeros((1, decoderUnitsnum))
    c = np.zeros((1, decoderUnitsnum))
    out = np.zeros((1, targetSentenceLen))
    idx = sentence2id(sentence=sourceSentence, word2id=sourceword2id, maxlen=sourceSentenceLen)
    idx = np.array(idx)

    pred = model.predict([idx.reshape(-1, 20), s, c, out])
    predictions = np.argmax(pred, axis=-1)
    idx = [sourceword2id.get(idx[0], '<UNK>') for idx in predictions]
    return ' '.join(idx)


def plotAttention(sentence, sourceSentenceLen, targetSentenceLen, sourceword2id, decoderUnitsnum, model):
    """
    :param sentence: 要翻译的源句子
    :param sourceSentenceLen: 源句子长度（padding后的长度）
    :param targetSentenceLen: padding后的长度
    :param sourceword2id: 源语言字典
    :param decoderUnitsnum: decoder中LSTM数量
    :param model: decoder中LSTM数量
    :return:
    """
    x = np.array(sentence2id(sentence, word2id=sourceword2id, maxlen=sourceSentenceLen, source=False))
    f = K.function(model.inputs,
                   [model.get_layer(name='attentionWeight').get_output_at(t) for t in range(targetSentenceLen)])
    s = np.zeros((1, decoderUnitsnum))
    c = np.zeros((1, decoderUnitsnum))
    out = np.zeros((1, targetSentenceLen))

    attention_alpha = f([x.reshape(-1, sourceSentenceLen), s, c, out])
    attention_matrix = np.zeros((targetSentenceLen, sourceSentenceLen))
    for j in range(targetSentenceLen):
        for i in range(sourceSentenceLen):
            attention_matrix[j][i] = attention_alpha[j][0, i, 0]
    y = prediction(sentence, sourceword2id=sourceword2id, sourceSentenceLen=sourceSentenceLen,
                   decoderUnitsnum=decoderUnitsnum, targetSentenceLen=targetSentenceLen, model=model)
    print('The translated sentence is {}'.format(y))
    sourceWordsList = sentence.split()
    targetWordsList = y.split()
    f, axes = plt.subplots(figsize=(15, 10))
    sns.heatmap(attention_matrix, xticklabels=sourceWordsList, yticklabels=targetWordsList, cmap='YlGnBu')
    axes.set_title('Attention Map\n(Target words vs Source words)', fontsize=14)
    plt.show()


if __name__ == '__main__':
    # 训练资料可到本文开头的Github地址去clone
    englishText_path = 'zhihu/mt_attention_birnn/data/small_vocab_en'
    frenchText_path = 'zhihu/mt_attention_birnn/data/small_vocab_fr'
    # glove采用哈佛大学100维的词向量, 下载地址可在网上找到镜像去下载，原网址下载速度太慢！
    glove_path = '/downloads/glove.6B/glove.6B.100d.txt'
    # encoder units
    encoderUnitsnum = 32
    # decoder units
    decoderUnitsnum = 128

    sourcetextidx, sourceword2id, sourceid2word, sourcedictsize, sourcesentencemaxlen = text2id(englishText_path,
                                                                                                source=True)
    targettextidx, targetword2id, targetid2word, targetdictsize, targetsentencemaxlen = text2id(frenchText_path,
                                                                                                source=False)
    embeddinglayer = pretrainedembedding(glove_path, sourceword2id, sourcesentencemaxlen)

    s = np.zeros((1, decoderUnitsnum))
    c = np.zeros((1, decoderUnitsnum))
    out = np.zeros((1, targetsentencemaxlen))

    x = sourcetextidx
    y = targettextidx
    # 将y转换为one-hot encoding, 模型最后计算categorical_crossentropy时要用到
    y = to_categorical(y, targetdictsize)
    y = list(y.swapaxes(0, 1))

    # 模型训练
    model = seq2seqwithAttention(sourceSentenceLen=sourcesentencemaxlen, targetSentenceLen=targetsentencemaxlen,
                                 encoderUnitsnum=encoderUnitsnum, decoderUnitsnum=decoderUnitsnum,
                                 sourceVocabsize=sourcedictsize, targetVocabsize=targetdictsize,
                                 embeddingLayer=embeddinglayer
                                 )
    model.fit(x=[x, s, c, out], y=y, epochs=5, batch_size=128)
    # 保存训练好的模型
    model.save('seq2seqWithAttention_{}.h5'.format(datetime.now().strftime('%Y-%M-%D')))

    test_sentence = 'she likes mangoes, apples and bananas.'
    plotAttention(test_sentence, sourceSentenceLen=sourcesentencemaxlen,
                  targetSentenceLen=targetsentencemaxlen, sourceword2id=sourceword2id,
                  decoderUnitsnum=decoderUnitsnum, model=model)


    # nltk 句子翻译得分计算 sentence_bleu
    # 如果想要看看训练的模型效果如何可以计算bleu得分
    sentence_bleu([["i", "like", "you", "."]], ["i", "love", "you", "."])







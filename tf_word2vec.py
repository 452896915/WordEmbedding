# -*- coding: utf-8 -*-

import urllib
import collections
import math
import os
import random
import zipfile
import datetime as dt

import numpy as np
import tensorflow as tf


def maybe_download(filename, url):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    #  [['UNK', -1], ['i', 500], ['the', 498], ['man', 312], ...]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    #  dictionary {'UNK':0, 'i':1, 'the': 2, 'man':3, ...}
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # data: "I like cat" -> [1, 21, 124]
    # count: [['UNK', 349], ['i', 500], ['the', 498], ['man', 312], ...]
    # dictionary {'UNK':0, 'i':1, 'the': 2, 'man':3, ...}
    # reversed_dictionary: {0:'UNK', 1:'i', 2:'the', 3:'man', ...}
    return data, count, dictionary, reversed_dictionary


def collect_data(vocabulary_size=10000):
    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('enwik8.zip', url)
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

data_index = 0

"""
generate batch data

data: "I like to watch a love movie..." -> [1, 21, 124, 438, 11, 434] 
batch_size: 128  
num_skips: 2  代表源单词左右两个方向扩展的单词范围
skip_window: 2  代表源单词的位置

思路：
从128个词开始的位置选5个单词，每次向后移动一个位置获取下5个单词。取5个单词的中间单词作为源单词，随机从生下的四个单词中取两个作为目标单词
这样就生成了两个source-target对。一共循环128/2 =64次，获取到64*2=128个source-target对，作为一个batch的训练数据。
"""
def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span含义 -> [ skip_window input_word skip_window ]

    # 初始化最大长度为span的双端队列，超过最大长度后再添加数据，会从另一端删除容不下的数据
    # buffer: 1, 21, 124, 438, 11
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):  # 128 / 2
        # target: 2
        target = skip_window  # input word at the center of the buffer
        # targets_to_avoid: [2]
        targets_to_avoid = [skip_window]  # 需要忽略的词在当前span的位置

        # 更新源单词为当前5个单词的中间单词
        source_word = buffer[skip_window]

        # 随机选择的5个span单词中除了源单词之外的4个单词中的两个
        for j in range(num_skips):
            while target in targets_to_avoid:  # 随机重新从5个词中选择一个尚未选择过的词
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)

            # batch添加源单词
            batch[i * num_skips + j] = source_word
            # context添加目标单词，单词来自随机选择的5个span单词中除了源单词之外的4个单词中的两个
            context[i * num_skips + j, 0] = buffer[target]

        # 往双端队列中添加下一个单词，双端队列会自动将容不下的数据从另一端删除
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context

vocabulary_size = 10000

# data: "I like cat" -> [1, 21, 124]
# count: [['UNK', 349], ['i', 500], ['the', 498], ['man', 312], ...]
# dictionary {'UNK':0, 'i':1, 'the': 2, 'man':3, ...}
# reversed_dictionary: {0:'UNK', 1:'i', 2:'the', 3:'man', ...}
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)

batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # 定义输入输出
  train_sources = tf.placeholder(tf.int32, shape=[batch_size])
  train_targets = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # 初始化embeddings矩阵,这个就是经过多步训练后最终我们需要的embedding
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

  # 将输入序列转换成embedding表示, [batch_size, embedding_size]
  embed = tf.nn.embedding_lookup(embeddings, train_sources)

  # 初始化权重
  weights = tf.Variable(tf.truncated_normal([embedding_size, vocabulary_size], stddev=1.0 / math.sqrt(embedding_size)))
  biases = tf.Variable(tf.zeros([vocabulary_size]))

  # 隐藏层输出结果的计算, [batch_size, vocabulary_size]
  hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

  # 将label结果转换成one-hot表示, [batch_size, 1] -> [batch_size, vocabulary_size]
  train_one_hot = tf.one_hot(train_targets, vocabulary_size)

  # 根据隐藏层输出结果和标记结果，计算交叉熵
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))

  # 随机梯度下降进行一步反向传递
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

  # 计算验证数据集中的单词和字典表里所有单词的相似度，并在validate过程输出相似度最高的几个单词
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # 参数初始化赋值
  init = tf.global_variables_initializer()


def run(graph, num_steps):
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data, batch_size, num_skips, skip_window)
        feed_dict = {train_sources: batch_inputs, train_targets: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      # 最终的embedding
      final_embeddings = normalized_embeddings.eval()

num_steps = 1000
softmax_start_time = dt.datetime.now()
run(graph, num_steps=num_steps)
softmax_end_time = dt.datetime.now()
print("Softmax method took {} seconds to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))

with graph.as_default():

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_targets,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 1000
nce_start_time = dt.datetime.now()
run(graph, num_steps)
nce_end_time = dt.datetime.now()
print("NCE method took {} seconds to run 100 iterations".format((nce_end_time-nce_start_time).total_seconds()))
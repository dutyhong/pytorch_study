import random

import pandas as pd
# waimai_comment_data = pd.read_csv("/Users/duty/desktop/waimai_comment.csv")
# n_classes = set(waimai_comment_data["label"])
# categories_statistics = waimai_comment_data.loc[:, 'label'].value_counts()
# positive_samples = waimai_comment_data.loc[waimai_comment_data['label']==1]
# # negative_samples = waimai_comment_data.loc[waimai_comment_data['label']==0]
# aa = '"ddddd"'
# bb = aa.replace('"', '')
aa  = range(1)
def get_train_test_data():
    # waimai_comment_data_file = open("/Users/duty/desktop/waimai_comment.csv", "r")
    waimai_comment_data_file = open("./waimai_comment.csv", "r")
    positive_samples = []
    negative_samples = []
    vocabs = set()
    for line in waimai_comment_data_file.readlines():
        columns = line.split(",")
        if len(columns)<2:
            break
        label = columns[0]
        text = columns[1].replace('"', "")
        line_vocabs = set(text)
        vocabs = vocabs|line_vocabs
        if label=='1':
            positive_samples.append((1, text))
        else:
            negative_samples.append((0, text))
    positive_cnt = len(positive_samples)
    negative_cnt = len(negative_samples)
    negative_samples_new = []
    for i in range(positive_cnt):
        negative_samples_new.append(negative_samples[i])
    train_cnt = 3000
    test_cnt = 500
    val_cnt = 500
    train_data = []
    test_data = []
    val_data = []
    for i, (positive_sample, negative_sample) in enumerate(zip(positive_samples, negative_samples_new)):
        if i<train_cnt:
            train_data.append(positive_sample)
            train_data.append(negative_sample)
        elif i>=train_cnt and i<(train_cnt+test_cnt):
            test_data.append(positive_sample)
            test_data.append(negative_sample)
        else:
            val_data.append(positive_sample)
            val_data.append(negative_sample)
    return train_data, test_data, val_data, vocabs

def get_robot_train_test_data():
    # cnews_data_file = open("/Users/duty/desktop/cnews.train.txt", "r")
    cnews_data_file = open("./cnews.train.txt", "r")
    total_samples = {}
    vocabs = set()
    labels = ["开口邀约", "承诺到店", "询问买什么车", "询问不买原因", "询问买车时间"]
    labels_ids = [0, 1,2,3,4]
    every_sample_cnt = 1000
    for line in cnews_data_file.readlines():
        columns = line.split("\t")
        if len(columns) < 2:
            break
        label = columns[0]
        text = columns[1].replace('"', "")
        line_vocabs = set(text)
        vocabs = vocabs | line_vocabs
        if label in total_samples:
            texts = total_samples[label]
            texts.append(text)
        else:
            total_samples[label] = [text]
    label_set = set()
    new_total_samples = {}
    for key, texts in total_samples.items():
        if key in labels:
            new_texts = texts[0:1000]
            new_total_samples[labels.index(key)] = new_texts

        # label_set.add(key)
        # print("%s 样本数 ：%d"%(key, len(texts)))
    ###将新的样本数据分位训练集和测试集
    ##训练集个数
    train_cnt = 800
    ##测试集个数
    test_cnt = 200
    train_data = []
    test_data = []
    for key, texts in new_total_samples.items():
        for i in range(train_cnt):
            train_data.append((key, texts[i]))
        for i in range(test_cnt):
            test_data.append((key, texts[train_cnt+i]))
    val_data = []
    random.shuffle(train_data)
    random.shuffle(test_data)
    random.shuffle(val_data)
    return train_data, test_data, val_data, vocabs

def get_batch_data(input_data, batch_size):
    data_cnt = len(input_data)
    batch_cnt = int(data_cnt/batch_size)
    batch_datas = []
    for i in range(batch_cnt):
        batch_data = []
        for j in range(batch_size):
            batch_data.append(input_data[i*batch_size+j])
        batch_datas.append(batch_data)
    batch_data = []
    if data_cnt-batch_cnt*batch_size == 0:
        return batch_datas
    else:
        for i in range(data_cnt-batch_cnt*batch_size):
            batch_data.append(input_data[batch_cnt*batch_size+i])
        batch_datas.append(batch_data)
    return batch_datas
# train_data, test_data, val_data, vocabs = get_train_test_data()
# batch_train_data = get_batch_data(train_data, 32)
labels = ["开口邀约", "承诺到店", "询问买什么车", "询问不买原因", "询问买车时间"]

get_robot_train_test_data()
print("ddd")
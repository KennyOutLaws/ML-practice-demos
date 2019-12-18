import numpy as np
from math import log
#calculate the shanno entropy
def calc_shanno_ent(data_set):
    num_entries = len(data_set)
    label_count ={}
    for feature_vec in data_set:
        current_label = data_set[-1]
        if current_label not in label_count.keys():
            label_count[current_label] =0
        label_count[current_label] = label_count[current_label]+1
    shanno_ent = 0
    for key in label_count.keys():
        prob = float(label_count[key] / num_entries)
        shanno_ent -=prob* log(prob,2)
    return shanno_ent
# split the dataset
def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feature in dataset:
        if feature[axis] == value:
            reduced_dataset = feature[:axis]
            reduced_dataset.extend(feature[axis+1:])
            ret_dataset.append(reduced_dataset)
    return ret_dataset
def choose_feature_to_split(dataset):
    num_features = len(dataset)-1
    base_ent = calc_shanno_ent(dataset)
    info_gain = 0
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        #very good job
        unique_value = set(feature_list)

        new_ent = 0.0
        for value in unique_value:
            split_data = split_dataset(dataset,i,value)
            prob = len(split_data)/len(dataset)
            new_ent += prob*calc_shanno_ent(split_data)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature
def majorityCnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] +=1
    sorted_class_count = sorted(class_count.items(), key= lambda x: x[1], reverse=True)
    # Debug and see why the index is [0][0]
    return sorted_class_count[0][0]

def createTree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # debug
    if len(dataset[0] == 1):
        majorityCnt(class_list)
    best_feature = choose_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in dataset]
    unique_feature = set(feature_values)
    for value in unique_feature:
        # prevent changing the original list by reference
        sub_labels = labels[:]
        tree[best_feature_label][value] = createTree(split_dataset(dataset, best_feature,value), sub_labels)

    return tree


def classify(input_tree, feature_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feature_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]) == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vec)
            else:
                class_label =  second_dict[key]
    return class_label
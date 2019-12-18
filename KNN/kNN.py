import numpy as np
import operator
def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX, (dataset_size,1)) - dataset
    diff_mat_square = diff_mat ** 2
    sq_distances = np.sum(diff_mat_square, axis=1)
    distances =  sq_distances ** 0.5
    sorted_distances = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances[i]]
        # Have some questions about it...
        class_count[vote_label] = class_count.get(vote_label,0) + 1
        # Take cautions over here, I changed it into python3 standard dict implementation...
    sorted_class_count = sorted(class_count.items(), key= lambda x: x[1],reverse=True)
    return sorted_class_count[0][0]
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    line_numbers = len(arrayOLines)
    return_mat = np.zeros((line_numbers,3))
    class_label_vector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        return_mat[index, :] = listFromLine[0:3]
        class_label_vector.append(int(listFromLine[-1]))
        index+=1
    return return_mat, class_label_vector
def autoNorm(dataset):
    min_value = dataset.min(0)
    max_value = dataset.max(0)
    ranges = max_value - min_value
    m = dataset.shape[0]
    norm_dataset = np.zeros(np.shape(dataset))
    norm_dataset = dataset - np.tile(min_value, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_value
def datingClassTest():
    ho_ratio = 0.1
    error = 0
    dating_data, dating_labels = file2matrix('datingTestSet2.txt')
    norm_dating, _,_ = autoNorm(dating_data)
    m = len(norm_dating)
    num_test_vec = int(ho_ratio * m)
    for i in range(num_test_vec):
        result = classify0(norm_dating[i],norm_dating[num_test_vec: -1,:]
                           ,dating_labels[num_test_vec:-1], 3)
        print('The classifier came back with: %d, the real answer is: %d'%(result, dating_labels[i]))

        if (result != dating_labels[i]):
            error +=1
    print('total error rate is %d'%(error/num_test_vec))
datingClassTest()
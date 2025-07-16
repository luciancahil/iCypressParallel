import torch
import numpy as np
import copy

def most_common(lst):
    return max(set(lst), key=lst.count)

file = open("./rawData/ESET_nodes.csv", mode='r')
labels = []
data = []

for line in file:
    line = line.strip()
    parts = line.split(',')
    designation = int(parts[0])

    info = np.array(parts[1:]).astype(np.float64)

    labels.append(designation)
    data.append(info)

num_data = len(data)
num_correct = 0
for i in range(num_data):
    cur_data = copy.deepcopy(data)
    cur_labels = copy.deepcopy(labels)
    actual = cur_labels.pop(i)
    test = cur_data.pop(i)
    places = torch.tensor(cur_data)
    test = torch.tensor(test)

    dist = torch.norm(places - test, dim=1, p=None)
    # change 3 for changes in K.
    knn = dist.topk(7, largest=False)
    votes = [cur_labels[i] for i in knn.indices]
    choice = most_common(votes)
    print('kNN dist: {}, index: {}, votes: {}, choice: {}, actual, {}'.format(knn.values, knn.indices, votes, choice, actual))

    if(choice == actual):
        num_correct += 1

print('Got {} correct out of {} ({}%)'.format(num_correct, num_data, (num_correct/num_data)))

# For 3:
# Got 225 correct out of 231 (0.974025974025974%)
# For 5: 
# Got 223 correct out of 231 (0.9653679653679653%)
# For 7: 
# Got 221 correct out of 231 (0.9567099567099567%)
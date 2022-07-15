import numpy
import scipy.io as scio
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from module import rgnn
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_data_list = []
test_data_list = []
labels = scio.loadmat('./seed/seed/label.mat')['label'][0]

all_mat_file = os.walk('./seed/seed/independent')
skip_set = {'lable.mat'}
feature_name = 'de_LDS'
label_noise = 0.8
num_node = 62
num_class = 3
epochs = 1000


def distance_square(location,i,j):
    return (location[i][0]-location[j][0])*(location[i][0]-location[j][0])+(location[i][1]-location[j][1])*(location[i][1]-location[j][1])


location = scio.loadmat('./location.mat')['location']
location = location.astype(np.float64)
'''

 FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ   
 0   1   2   3   4   5  6  7  8  9

 F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2  
 10 11 12 13 14  15  16  17  18  19

 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4      
 20  21  22  23 24 25 26 27 28 29

 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6   
 30 31 32  33  34  35  36  37  38  39

 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8    
 40  41 42 43 44 45 46 47 48 49

 PO7 PO5 PO3 POZ PO4 PO6 PO8 CB1 O1 OZ   
 50  51  52  53  54  55  56  57  58 59

 O2 CB2  
 60 61

'''



A = numpy.zeros((num_node,num_node))

for i in range(num_node):
    for j in range(num_node):
        k = distance_square(location,i,j)
        np.seterr(divide='ignore', invalid='ignore')
        A[i][j] = min(1,5/k)
A = torch.from_numpy(A)



A[0][2]=A[2][0]=A[0][2]-1
A[3][4]=A[4][3]=A[3][4]-1
A[6][12]=A[12][6]=A[6][12]-1
A[15][21]=A[21][15]=A[15][21]-1
A[24][30]=A[30][24]=A[24][30]-1
A[33][39]=A[39][33]=A[33][39]-1
A[42][48]=A[48][42]=A[42][48]-1
A[51][55]=A[55][51]=A[51][55]-1
A[58][60]=A[60][58]=A[58][60]-1



node_out = []
node_in = []
for i in range(num_node):
    for j in range(num_node-i):
        node_out.append(i)
        node_in.append(num_node-j-1)
node_in = np.array(node_in)
node_out = np.array(node_out)
edge_index = np.vstack((node_out,node_in))
edge_index = torch.from_numpy(edge_index)
edge_index = edge_index.long()


edge_weight = []
num_edge = np.size(edge_index,1)
for i in range(num_edge):
    edge_weight.append(A[node_out[i]][node_in[i]])
edge_weight = np.array(edge_weight)
edge_weight = torch.from_numpy(edge_weight)
edge_weight = torch.abs(edge_weight)
edge_weight = edge_weight.float()



for path,dir_list,file_list in all_mat_file:
    train_list = file_list[:9]
    test_list = file_list[9:]
    # trian set
    for file_name in train_list:
        if file_name not in skip_set:
            all_features_dict = scio.loadmat(os.path.join('./seed/seed/independent',file_name))
            subject_name = file_name.split('.')[0]
            for trials in range(1, 16):
                cur_feature = all_features_dict[feature_name + str(trials)]
                cur_feature = np.asarray(cur_feature[:, 0:180, :])

                cur_num_node = np.size(cur_feature, 0)
                cur_feature = np.reshape(cur_feature, (cur_num_node, -1))
                # test
                for i in range(np.size(cur_feature, 0)):
                    for j in range(np.size(cur_feature, 1)):
                        cur_feature[i][j] = cur_feature[i][j] / 30

                cur_feature = torch.from_numpy(cur_feature)
                cur_feature = cur_feature.float()
                cur_label = labels[trials - 1]
                if cur_label == -1:
                    cur_label = [1 - 2 * label_noise / 3, 2 * label_noise / 3, 0]
                elif cur_label == 0:
                    cur_label = [label_noise / 3, 1 - 2 * label_noise / 3, label_noise / 3]
                elif cur_label == 1:
                    cur_label = [0, 2 * label_noise / 3, 1 - 2 * label_noise / 3]
                data = Data(x=cur_feature, edge_index=edge_index, edge_attr=edge_weight, y=cur_label)
                train_data_list.append(data)

    # test set
    for file_name in test_list:
        if file_name not in skip_set:
            all_features_dict = scio.loadmat(os.path.join('./seed/seed/independent', file_name))
            subject_name = file_name.split('.')[0]

            for trials in range(1, 16):
                cur_feature = all_features_dict[feature_name + str(trials)]
                cur_feature = np.asarray(cur_feature[:, 0:180, :])

                cur_num_node = np.size(cur_feature, 0)
                cur_feature = np.reshape(cur_feature, (cur_num_node, -1))
                # test
                for i in range(np.size(cur_feature, 0)):
                    for j in range(np.size(cur_feature, 1)):
                        cur_feature[i][j] = cur_feature[i][j] / 30

                cur_feature = torch.from_numpy(cur_feature)
                cur_feature = cur_feature.float()
                cur_label = labels[trials - 1]
                if cur_label == -1:
                    cur_label = [1 - 2 * label_noise / 3, 2 * label_noise / 3, 0]
                elif cur_label == 0:
                    cur_label = [label_noise / 3, 1 - 2 * label_noise / 3, label_noise / 3]
                elif cur_label == 1:
                    cur_label = [0, 2 * label_noise / 3, 1 - 2 * label_noise / 3]
                data = Data(x=cur_feature, edge_index=edge_index, edge_attr=edge_weight, y=cur_label)
                test_data_list.append(data)

batch_size = 16
train_set = DataLoader(train_data_list, batch_size=batch_size)
test_set = DataLoader(train_data_list, batch_size=batch_size)


num_in = train_data_list[0].x.size(1)
net = rgnn(num_in=num_in, num_hidden=30, K=2, num_class=num_class, dropout=0.7, domain_adaptation=1,
           weight=edge_weight)
net.to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-5)


def evaluation(dataloader):
    total, correct = 0, 0
    for data in dataloader:
        x = data.x
        x = x.to(device)
        index = data.edge_index
        index = index.to(device)
        weight = data.edge_attr
        weight = weight.to(device)
        batch = data.batch
        batch = batch.to(device)
        len_y = len(data.y)
        y = data.y
        y = torch.Tensor(data.y)
        y = y.to(device)
        pre, _ = net(len_y, x, index, batch)

        total += len_y

        for i in range(y.size(0)):
            if torch.abs(y[i][0]-pre[i][0])>0.25:
                continue
            if torch.abs(y[i][1]-pre[i][1])>0.25:
                continue
            if torch.abs(y[i][2]-pre[i][2])>0.25:
                continue
            correct = correct+1
    return 100 * correct / total

image_x = []
image_y = []
#  train
for epoch in range(100):
    loss_nn = 0
    batch_nn = 0
    for data in train_set:
        batch_int = data.batch.size(0)
        train_domain = torch.zeros((batch_int))
        train_domain = train_domain.long()
        train_domain = train_domain.to(device)

        x = data.x
        x = x.to(device)
        index = data.edge_index
        index = index.to(device)
        weight = data.edge_attr
        weight = weight.to(device)
        batch = data.batch
        batch = batch.to(device)
        len_y = len(data.y)
        y = data.y
        y = torch.Tensor(data.y)
        y = y.to(device)

        opt.zero_grad()
        label, domain = net(len_y, x, index, batch)

        loss0 = nn.CrossEntropyLoss()
        loss1 = loss0(domain, train_domain)
        loss2 = F.kl_div(label.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
        loss = loss1 + loss2

        loss_nn = loss_nn + loss
        batch_nn = batch_nn + 1

        loss.backward()
        opt.step()
    for data in test_set:
        batch_int = data.batch.size(0)
        train_domain = torch.ones((batch_int))
        train_domain = train_domain.long()
        train_domain = train_domain.to(device)

        x = data.x
        x = x.to(device)
        index = data.edge_index
        index = index.to(device)
        weight = data.edge_attr
        weight = weight.to(device)
        batch = data.batch
        batch = batch.to(device)
        len_y = len(data.y)
        y = data.y
        y = torch.Tensor(data.y)
        y = y.to(device)

        opt.zero_grad()
        label, domain = net(len_y, x, index, batch)

        loss0 = nn.CrossEntropyLoss()
        loss1 = loss0(domain, train_domain)
        loss2 = F.kl_div(label.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
        loss = loss1 + loss2

        loss_nn = loss_nn + loss
        batch_nn = batch_nn + 1

        loss.backward()
        opt.step()
    image_x.append(100 + epoch)
    image_y.append((loss_nn / batch_nn).item())
    print('Epoch: %d/%d, loss: %0.2f' % (
        epoch, epochs, loss_nn / batch_nn))
    print('Epoch: %d/%d, Train acc: %0.2f' % (
        epoch, epochs, evaluation(train_set)))

plt.plot(image_x,image_y)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("./image/test002.png")
plt.show()

print(' Test acc: %0.2f' % (evaluation(test_set)))
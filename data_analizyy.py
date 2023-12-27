import numpy as np
import torch
import pandas as pd
from data.data_access import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import seaborn as sns
ep_number = 1
epeach = 1000
dropouts = [0.99]
momentums = [0.99]
weight_decays = [5e-3]
batchs = [20]
lrs = [1e-5]
first_channels = [2000]
second_channels = [1900]
return_number = 5
return_spilit = 60


def seed_everything(seed):
    '''

    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


seed_everything(3407)

data_pa = {'id': 'ALL', 'type': 'prostate_paper',
           'params': {'data_type': ['mut_important', 'cnv_del', 'cnv_amp'], 'drop_AR': False, 'cnv_levels': 3,
                      'mut_binary': True, 'balanced_data': False, 'combine_type': 'union',
                      'use_coding_genes_only': True,
                      'selected_genes': 'tcga_prostate_expressed_genes_and_cancer_genes.csv', 'training_split': 0}}
data = Data(**data_pa)
x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

x_t = np.concatenate((x_test_, x_validate_))
y_t = np.concatenate((y_test_, y_validate_))
train=[]

def caculateAUC(AUC_outs, AUC_labels):
    ROC = 0
    outs = []
    labels = []
    for (index, AUC_out) in enumerate(AUC_outs):
        softmax = nn.Softmax(dim=1)
        out = softmax(AUC_out).numpy()
        out = out[:, 1]
        for out_one in out.tolist():
            outs.append(out_one)
        for AUC_one in AUC_labels[index].tolist():
            labels.append(AUC_one)

    outs = np.array(outs)

    labels = np.array(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, outs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, outs)

    return auc, aupr

# print(x_train)
# print("......................")
# x_train=x_train.reshape(-1,9229,3)
# print(x_train)


x_train = x_train.reshape( -1,9229, 3)
print(x_train.shape)
class DCSN(nn.Module):

    def __init__(self):
        super(DCSN, self).__init__()
        self.W_q = nn.Linear(2, 2)
        self.W_k = nn.Linear(2, 2)
        self.W_v = nn.Linear(2, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv1d(9229, 2000, kernel_size=(3,), stride=1)
        nn.init.xavier_normal_(self.conv1.weight, gain=1)

        self.conv2 = nn.Conv1d(9229, 1900, kernel_size=(3,), stride=1)
        nn.init.xavier_normal_(self.conv2.weight, gain=1)

        self.Flatten = nn.Flatten()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2000 + 1900, 2)
        nn.init.xavier_normal_(self.linear1.weight, gain=1)

        # self.linear3=nn.Linear(100,2)
        self.BatchNorm1 = nn.BatchNorm1d(num_features=2000)

        self.BatchNorm2 = nn.BatchNorm1d(num_features=1900)
        self.dropout = nn.Dropout(0.98)

    def forward(self, input):
        # food_conv1 = self.maxpool1(self.dropout(self.BatchNorm(self.relu(self.food_conv1(input)))).squeeze(3))
        input = input.reshape(-1, 9229 * 3)

        input = input.reshape(-1, 9229, 3)
        conv1 = self.conv1(input)
        conv1 = self.relu(conv1)
        conv1 = self.BatchNorm1(conv1)
        conv1 = self.dropout(conv1)

        #

        conv2 = self.conv2(input)
        conv2 = self.relu(conv2)
        conv2 = self.BatchNorm2(conv2)
        conv2 = self.dropout(conv2)

        all = torch.cat([conv1, conv2], 1).squeeze(2)

        all = self.linear1(all)
        # np.save("l_w", self.linear1.weight.data.cpu().detach().numpy())
        # np.save("c1_w", self.conv1.weight.data.cpu().detach().numpy())
        # np.save("c2_w", self.conv2.weight.data.cpu().detach().numpy())

        return all

model = DCSN()
model.load_state_dict(torch.load("model.ckpt",map_location=torch.device('cpu')))
model.eval()
# print(model.BatchNorm1(model.relu(model.conv1(torch.from_numpy(x_train).float()))))

c1_w=model.conv1.weight.data.numpy()
print(c1_w.shape)
c2_w=model.conv2.weight.data.numpy()
print(c2_w.shape)
l_w=model.linear1.weight.data.numpy()
print(l_w.shape)
gene_p=[]
for x in x_train:
    # for i in x:
    #     print(i)
    c1_o_channel=[]
    for c_k in c1_w:
        input_channel=[]
        for i,c1 in enumerate(c_k):
            input_channel.append(x[i][0]*c1[0]+x[i][1]*c1[1]+x[i][2]*c1[2])
        c1_o_channel.append(input_channel)
    c2_o_channel=[]
    for c_k in c2_w:
        input_channel=[]
        for i,c1 in enumerate(c_k):
            input_channel.append(x[i][0]*c1[0]+x[i][1]*c1[1]+x[i][2]*c1[2])
        c2_o_channel.append(input_channel)
    c1_o_channel=np.array(c1_o_channel)
    c1_o_channel_sum=c1_o_channel.sum(-1)
    c1_abs_sum=np.abs(c1_o_channel).sum(-1)
    c2_o_channel=np.array(c2_o_channel)
    c2_o_channel_sum=c2_o_channel.sum(-1)
    c2_abs_sum=np.abs(c2_o_channel).sum(-1)
    o_channel=np.concatenate([c1_o_channel,c2_o_channel],axis=0)

    print(o_channel.shape)
    o_channel_sum = np.concatenate([c1_o_channel_sum, c2_o_channel_sum], axis=0)
    o_abs_sum = np.concatenate([c1_abs_sum, c2_abs_sum], axis=0)
    print(o_channel_sum.shape)
    b1=model.BatchNorm1(model.relu(torch.from_numpy(c1_o_channel_sum).reshape(1,-1,1).float())).detach().numpy()
    b2=model.BatchNorm2(model.relu(torch.from_numpy(c2_o_channel_sum).reshape(1,-1,1).float())).detach().numpy()
    b_sum=np.concatenate([b1.reshape(2000),b2.reshape(1900)],axis=0)
    print(b_sum.shape)
    class_hot=[]
    for i,w in enumerate(b_sum):

        class_hot.append([w*l_w[0][i],w*l_w[1][i]])
    class_hot=np.array(class_hot)
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(class_hot)
    # plt.xlabel('Classification')
    # plt.ylabel('Instance Index')
    # plt.savefig("classification_heatmap.jpg")
    # plt.show()
    l_one_p=class_hot[:,1]/np.abs(class_hot).sum(0)[1]
    print(class_hot.sum(0)[1])
    for i,p in enumerate(l_one_p):
        minus=False
        if p>0 and b_sum[i]<0:
            minus=True
        elif p<0 and b_sum[i]>0:
            minus = True
        sum_w=o_abs_sum[i]
        if minus:
            sum_w=-sum_w
        # print(o_channel[i])
        # print(sum_w)
        o_channel[i]=(o_channel[i]/sum_w)*abs(p)
    this_gene_p=o_channel.sum(0)
    print(this_gene_p.shape)
    gene_p_arrary=[]

    for p in this_gene_p:
        gene_p_arrary.append([p])
    gene_p.append(gene_p_arrary)
    this_gene_p=np.array(gene_p_arrary)
    print(this_gene_p)
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(np.array(this_gene_p))
    # plt.xlabel('Classification')
    # plt.ylabel('Instance Index')
    # plt.savefig("classification_heatmap.jpg")
    # plt.show()
gene_p=np.array(gene_p)
np.save("gene_p",gene_p)










import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
order=np.load("order.npy",allow_pickle=True)
gene_p=np.load("gene_p.npy")
gene_p=gene_p.sum(0).reshape(9229) #1939 6128   87
for g in order[np.argsort(-gene_p)]:
    print(g)
# plt.figure(figsize=(10, 10))
# sns.heatmap(np.array(gene_p))
# plt.xlabel('Classification')
# plt.ylabel('Instance Index')
# plt.savefig("classification_heatmap.jpg")
# plt.show()
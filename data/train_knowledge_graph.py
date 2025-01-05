import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# STEP 3
# 1. 加载 typhoon_windfarm_relationships.csv
file_path = 'typhoon_windfarm_relationships.csv'
data = pd.read_csv(file_path)

# 提取三元组
triplets = data[['Entity1', 'Relationship', 'Entity2']].values.tolist()

# 2. 创建实体和关系的映射
# 将所有实体和关系编码为索引
entities = set(data['Entity1']).union(set(data['Entity2']))
relations = set(data['Relationship'])
entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {relation: idx for idx, relation in enumerate(relations)}

# 将三元组转为索引形式
indexed_triplets = [
    [entity2id[head], relation2id[relation], entity2id[tail]]
    for head, relation, tail in triplets
]

# 转换为 Tensor
triplets_tensor = torch.tensor(indexed_triplets, dtype=torch.long)


# 3. 生成负样本
def generate_negative_samples(triplets_tensor, num_entities):
    """
    生成负样本，通过随机替换头实体或尾实体。
    """
    negative_triplets = triplets_tensor.clone()
    for i in range(negative_triplets.size(0)):
        if torch.rand(1).item() > 0.5:  # 随机替换头实体
            new_head = torch.randint(0, num_entities, (1,))
            while new_head == negative_triplets[i, 0]:  # 确保新实体不同于原实体
                new_head = torch.randint(0, num_entities, (1,))
            negative_triplets[i, 0] = new_head
        else:  # 随机替换尾实体
            new_tail = torch.randint(0, num_entities, (1,))
            while new_tail == negative_triplets[i, 2]:  # 确保新实体不同于原实体
                new_tail = torch.randint(0, num_entities, (1,))
            negative_triplets[i, 2] = new_tail
    return negative_triplets


# 4. 数据集拆分
train_triplets, test_triplets = train_test_split(triplets_tensor, test_size=0.2, random_state=42)


# 5. 使用 TransEModel 进行训练
class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0, p=1):
        super(TransEModel, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.p = p

        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

    def forward(self, positive_triplets, negative_triplets):
        pos_distance = self._distance(positive_triplets)
        neg_distance = self._distance(negative_triplets)
        return torch.relu(self.margin + pos_distance - neg_distance).mean()

    def _distance(self, triplets):
        head = self.entity_embeddings(triplets[:, 0])
        relation = self.relation_embeddings(triplets[:, 1])
        tail = self.entity_embeddings(triplets[:, 2])
        return torch.norm(head + relation - tail, p=self.p, dim=1)


# 参数设置
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
num_entities = len(entity2id)
num_relations = len(relation2id)
embedding_dim = 10
margin = 1.0

# 模型实例
model = TransEModel(num_entities, num_relations, embedding_dim, margin)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练步骤
epochs = 10
batch_size = 32
patience = 3  # 早停机制的耐心次数
best_loss = float('inf')
no_improvement = 0  # 记录连续未改善的次数

for epoch in range(epochs):
    model.train()
    total_loss = 0

    # 按批次训练
    for i in range(0, len(train_triplets), batch_size):
        positive_batch = train_triplets[i:i + batch_size]
        negative_batch = generate_negative_samples(positive_batch, num_entities)

        optimizer.zero_grad()
        loss = model(positive_batch, negative_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 计算每个epoch的平均loss
    average_loss = total_loss / len(train_triplets)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss}")

    # 检查是否早停
    if average_loss < best_loss:
        best_loss = average_loss
        no_improvement = 0  # 重置计数器
        torch.save(model.state_dict(), 'best_transe_model.pth')  # 保存最优模型
    else:
        no_improvement += 1
        print(f"No improvement for {no_improvement} epochs.")

    if no_improvement >= patience:
        print("Early stopping triggered.")
        break

# 保存模型
torch.save(model.state_dict(), 'transe_model.pth')

# 保存映射
import pickle

with open('entity2id.pkl', 'wb') as f:
    pickle.dump(entity2id, f)
with open('relation2id.pkl', 'wb') as f:
    pickle.dump(relation2id, f)

# 测试
model.eval()
with torch.no_grad():
    positive_test = test_triplets
    negative_test = generate_negative_samples(test_triplets, num_entities)
    test_loss = model(positive_test, negative_test)
    print(f"Test Loss: {test_loss.item()}")

# 加载self.entity_embeddings
entity_embeddings = model.entity_embeddings.weight.data.numpy()
embedding = entity_embeddings[5]
print(embedding.shape)


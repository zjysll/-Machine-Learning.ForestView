Transformer结合随机森林时序预测，由于数据集不公开，自己的数据集并没有很好的处理效果，根据对数据NAN处理方法不同，环境影响因子对反硝化潜势Feature important结果差别不小。
因此选择随机生成数据集，对两个模型结合进行探索，对相应的我以后需要用到的数据分析方法，以及相应的特征图。
用Deeseek和小浣熊AI进行指令生成代码，进行我所需图片的相应分析进行初步探索。所有内容均是原创

2. 数据集拆分与窗口化
from sklearn.model_selection import train_test_split

# 定义滑动窗口函数
def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])
        y.append(data[(i + 1):(i + window_size + 1), :])  # 预测整个下一个窗口
    return np.array(X), np.array(y)

# 设置窗口大小
window_size = 10

# 创建滑动窗口数据
X, y = create_sliding_window(scaled_data[['Sine', 'Cosine', 'Noisy']].values, window_size)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3. Transformer 回归模型
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        # src的形状应该是 [window_size, batch_size, input_dim]
        output = self.transformer_encoder(src)
        # 对整个序列进行预测
        output = self.fc(output)
        return output

# 模型参数
input_dim = 3
output_dim = 3
nhead = 1
num_encoder_layers = 2
dim_feedforward = 512

# 初始化模型
model = TransformerModel(input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward)

4.模型训练
import torch.optim as optim

# 转换数据为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 输入数据的维度调整为 [window_size, batch_size, input_dim] 以符合Transformer的输入要求
    outputs = model(X_train_tensor.permute(1, 0, 2))
    
    # 确保输出是三维的，然后调整维度顺序
    outputs = outputs.permute(1, 0, 2)
    
    # 计算损失，确保输出和目标的形状匹配 [batch_size, window_size, output_dim]
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Transformer Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

9.随机森林特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# 打印特征重要性
print("Feature importances:")
for i in range(X_train_rf.shape[1]):
    print(f"{i+1}. feature {indices[i]} ({importances[indices[i]]})")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train_rf.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_rf.shape[1]), indices)
plt.xlim([-1, X_train_rf.shape[1]])
plt.show()

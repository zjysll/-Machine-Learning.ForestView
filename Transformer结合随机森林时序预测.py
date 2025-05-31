Transformer结合随机森林时序预测，由于数据集不公开，自己的数据集并没有很好的处理效果，根据对数据NAN处理方法不同，环境影响因子对反硝化潜势Feature important结果差别不小。
因此选择随机生成数据集，对两个模型结合进行探索，对相应的我以后需要用到的数据分析方法，以及相应的特征图。
用Deeseek和小浣熊AI进行指令生成代码，进行我所需图片的相应分析进行初步探索。所有内容均是原创

1. 数据集生成以及预处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# 设置随机种子以保证结果可重复
np.random.seed(42)
# 生成时间序列数据
time_steps = np.arange(0, 100, 0.1)
#从0开始，到100结束，不包括100，步长为0.1
sine_wave = np.sin(time_steps)
#计算time_steps中每个时间点的正弦值
cosine_wave = np.cos(time_steps)
#计算time_steps中每个时间点的余弦值
noisy_signal = np.random.normal(0, 1, len(time_steps)) + 0.5 * np.sin(2 * time_steps)
#这行代码生成了一个长度为 1000 的正态分布随机数数组，表示噪声

# 创建数据集
data = pd.DataFrame({
'Time': time_steps,
'Sine': sine_wave,
'Cosine': cosine_wave,
'Noisy': noisy_signal
})

# 数据标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[[ 'Sine', 'Cosine', 'Noisy']])
scaled_data = pd.DataFrame(scaled_data, columns=[ 'Sine', 'Cosine', 'Noisy']) scaled_data[ 'Time'] = data[ 'Time']

# 可视化标准化后的数据
plt.figure(figsize=(14, 8))
# 设置绘图窗口大小为宽14英寸、高8英寸
plt.plot(scaled_data[ 'Time'], scaled_data[ 'Sine'], label= 'Normalized Sine Wave')
# 绘制标准化后的正弦波
plt.plot(scaled_data[ 'Time'], scaled_data[ 'Cosine'], label= 'Normalized Cosine Wave')
# 绘制标准化后的余弦波
plt.plot(scaled_data[ 'Time'], scaled_data[ 'Noisy'], label= 'Normalized Noisy Signal') 
# 绘制标准化后的带噪声信号
plt.title( 'Normalized Multivariate Time Series Data')
# 添加图表标题
plt.xlabel( 'Time')
# 设置x轴标签为“Time”
plt.ylabel( 'Normalized Value')
# 设置y轴标签为“Normalized Value”
plt.legend() 
plt.show()

2. 数据集拆分与窗口化
from sklearn.model_selection import train_test_split
#导入用于将数据集拆分为训练集和测试集的函数

# 定义滑动窗口函数
def create_sliding_window(data, window_size):
    X, y = [], []
#定义一个函数 create_sliding_window，用于将时间序列按指定窗口大小切分，初始化两个空列表 X 和 y，分别存放输入序列和目标序列。
    for i in range(len(data) - window_size):
#循环遍历数据，范围是从 0 到 len(data) - window_size，避免越界。
        X.append(data[i:(i + window_size), :])
#每次取一个窗口的长度作为输入数据 X。
        y.append(data[(i + 1):(i + window_size + 1), :])  # 预测整个下一个窗口
    return np.array(X), np.array(y)
#将 X 和 y 转换为 NumPy 数组返回。


# 设置窗口大小
window_size = 10
#定义滑动窗口大小为10。

# 创建滑动窗口数据
X, y = create_sliding_window(scaled_data[['Sine', 'Cosine', 'Noisy']].values, window_size)
#从 scaled_data 中提取三列，创建输入 (X) 和目标 (y) 数据。

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#将数据集按 80/20 拆分为训练集和测试集。

3. Transformer 回归模型
import torch
import torch.nn as nn
#导入 PyTorch 库以及神经网络模块。

class TransformerModel(nn.Module):
#定义一个名为 TransformerModel 的类，继承自 nn.Module。
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
#初始化方法：接收输入输出维度、多头数量、层数等参数，并调用父类构造函数。
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
#创建 Transformer 编码器，使用多个编码器层（TransformerEncoderLayer）堆叠而成。
        self.fc = nn.Linear(input_dim, output_dim)
#全连接层，用于将 Transformer 的输出映射到期望的输出维度。

    def forward(self, src):
        # src的形状应该是 [window_size, batch_size, input_dim]
        output = self.transformer_encoder(src)
        # 对整个序列进行预测
        output = self.fc(output)
        return output
#前向传播：输入 src shape 为 [window_size, batch_size, input_dim]，Transformer 输出后通过 fc 映射至 output_dim

# 模型参数
input_dim = 3
output_dim = 3
nhead = 1
num_encoder_layers = 2
dim_feedforward = 512
#设置 Transformer 模型参数：输入维度和输出维度均为 3（因为有 3 个特征），注意力头数为 1，编码器层数为 2，前馈层维度为 512

# 初始化模型
model = TransformerModel(input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward)
#设置模型参数并实例化模型

4.模型训练
import torch.optim as optim
#导入优化器模块。

# 转换数据为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#将 NumPy 数组转换为 PyTorch 张量，以便模型训练。

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#损失函数：均方误差（回归任务），优化器：Adam，自适应学习率。

num_epochs = 20
losses = []
#设置训练轮数，记录损失。

for epoch in range(num_epochs):
    model.train()
    # 开始训练循环，设置模型为训练模式。
    optimizer.zero_grad()
    #清除前一次迭代中累积的梯度。
    
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
#训练每一轮：调整输入维度以符合 Transformer 的 [seq_len, batch, dim] 格式；输出重新转回 [batch, seq_len, dim]；计算损失、反向传播、更新权重。

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Transformer Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#绘制训练损失随 epoch 变化的折线图，用于观察模型训练效果。

9.随机森林特征重要性
importances = rf_model.feature_importances_
#rf_model 是训练好的随机森林模型，.feature_importances_ 是随机森林提供的属性，表示每个特征对模型预测的贡献（按 Gini 或信息增益分数计算）。
indices = np.argsort(importances)[::-1]
#对特征重要性从大到小排序：np.argsort(importances)：返回按从小到大的排序索引，[::-1]：翻转索引，实现降序排列。结果：最重要的特征排在最前面。

# 打印特征重要性
print("Feature importances:")
#打印标题，提示接下来将输出特征重要性列表。
for i in range(X_train_rf.shape[1]):
    #遍历所有特征的索引，X_train_rf.shape[1] 表示特征总数。
    print(f"{i+1}. feature {indices[i]} ({importances[indices[i]]})")
    #按重要性排序打印每个特征的信息indices[i]：第 i 名重要的特征索引，importances[indices[i]]：对应的特征重要性分数。

# 可视化特征重要性
plt.figure(figsize=(10, 6))
#新建一个图像，设置图像大小为 10 x 6 英寸。
plt.title("Feature Importances")
#设置图表标题。
plt.bar(range(X_train_rf.shape[1]), importances[indices], align="center")
画出条形图：横坐标为特征索引（按重要性排序），纵坐标为对应的重要性分数。
plt.xticks(range(X_train_rf.shape[1]), indices)
#设置 x 轴刻度标签为实际特征索引（从大到小排序的）。
plt.xlim([-1, X_train_rf.shape[1]])
#设置 x 轴显示范围，-1 到特征数，防止边界数据被截断。
plt.show()
#显示图像。

10.融合模型误差分析
errors = y_test_rf -  ensemble_predictions
#计算误差值
plt.figure(figsize=(10, 6))
# 设置绘图窗口大小为宽10英寸、高6英寸
plt.hist(errors.flatten(), bins=50, alpha=0.7, color='blue') 
# 绘制误差分布的直方图
plt.title( 'Error Distribution of Ensemble Model')
# 添加图表标题
plt.xlabel( 'Error')
# 设置x轴标签为“Error”
plt.ylabel('Frequency') 
# 设置y轴标签为“Frequency”
plt.show()

11.残差图
# 计算残差
residuals = y_test_rf -  ensemble_predictions
# 绘制残差图
plt.figure(figsize=(10, 6))
# 设置绘图窗口大小为宽10英寸、高6英寸
plt.scatter(ensemble_predictions, residuals, alpha=0.5) 
# 绘制散点图，展示预测值与残差的关系
plt.axhline(y=0, color= 'r', linestyle='--')
# 添加一条水平线，表示残差为 0 的位置
plt.title( 'Residual Plot')
# 添加图表标题
plt.xlabel( 'Predicted Values')
# 设置 x 轴标签为“Predicted Values”
plt.ylabel( 'Residuals')
# 设置 y 轴标签为“Residuals”
plt.show()

12.预测Vs真实值
# 绘制预测 vs 真实值图
plt.figure(figsize=(10, 6))
# 设置绘图窗口大小为宽10英寸、高6英寸
plt.scatter(y_test_rf, ensemble_predictions, alpha=0.5)
# 绘制散点图，展示真实值与预测值的关系
plt.plot([y_test_rf.min(), y_test_rf.max()], [y_test_rf.min(), y_test_rf.max()], 'k-- ', lw=2) 
# 添加一条对角线，表示完美预测的情况
plt.title( 'Predicted vs Actual Plot')
# 添加图表标题
plt.xlabel( 'Actual Values')
# 设置 x 轴标签为“Actual Values”
plt.ylabel( 'Predicted Values')
# 设置 y 轴标签为“Predicted Values”
plt.show()


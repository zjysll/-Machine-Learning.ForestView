本研究旨在基于随机森林算法构建环境驱动因子解析模型，重点探讨土壤pH值、温度梯度、有机碳含量及含水率等关键参数对反硝化作用动力学特征（以潜在反硝化速率为表征）以及氧化亚氮（N2O）排放通量的影响机制，对相关驱动因子的feature important进行分析
鉴于原始农业环境数据存在异质性较高、信噪比不足等局限性，导致非线性模型的预测效能未能显著优于传统线性回归方法，本研究转而采用scikit-learn机器学习库中的乳腺癌数据集进行方法学验证。该替代数据集可有效模拟多变量交互作用下的特征解析过程，后续将系统阐述各算法组件的功能实现及其生态学意义解析方法。
郑鑫、刘英杰、黄梦甜、祝婧怡构造运营环境，下载nump包、pandas包时用到了Anconda prompt
郑鑫进行代码复刻时用到了Vs code
祝婧怡、黄梦甜、刘英杰在编码实现过程中，通过人工智能驱动的方法论，借助AI软件如DeepSeek、python等对多维数据进行建模与特征解析，并采用特征重要性量化分析框架实现驱动因子的动态权重评估。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler


# 1. 数据加载与预处理
# 载入乳腺癌数据集，该数据集包含 569 个样本，每个样本有 30 个特征和二分类标签（良性/恶性）
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 输出数据集基本信息
print("数据集特征形状: ", X.shape)
print("数据集标签分布:\n", y.value_counts())

# 数据预处理：标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将标准化后的数据转换为DataFrame便于后续处理和可视化
X_scaled = pd.DataFrame(X_scaled, columns=data.feature_names)

# 2. 数据探索性分析（Exploratory Data Analysis, EDA）
# 分析数据集的基本统计量、相关性和分布情况，帮助我们更好地理解数据
print("\n数据集描述统计信息:\n", X.describe())

# 可视化数据特征间的相关性热力图
plt.figure(figsize=(14, 10))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r")
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.xlabel("X Axis", fontsize=14)
plt.ylabel("Y Axis", fontsize=14)
plt.tight_layout()
plt.show()

# 可视化标签分布情况：良性与恶性样本比例
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="bright")
plt.title("Label Distribution", fontsize=16)
plt.xlabel("X Axis", fontsize=14)
plt.ylabel("Y Axis", fontsize=14)
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.tight_layout()
plt.show()

# 3. 划分训练集和测试集
# 为了评估模型的泛化能力，将数据集随机分为训练集和测试集，其中测试集占比 30%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
print("\n训练集样本数: ", X_train.shape[0])
print("测试集样本数: ", X_test.shape[0])

# 4. 构建基础随机森林分类器
# 初步构建随机森林模型，并进行训练和预测
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 输出分类报告和混淆矩阵
print("\nInitial Model Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.tight_layout()
plt.show()

# 5. 随机森林超参数调优
# 通过 GridSearchCV 网格搜索方法寻找最佳超参数组合
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# 采用 5 折交叉验证
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                           scoring='accuracy')

# 训练调优
grid_search.fit(X_train, y_train)

# 输出最佳超参数及最佳得分
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Accuracy: {:.4f}".format(grid_search.best_score_))

# 6. 使用最优参数构建优化后的随机森林模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)

# 输出优化后模型的分类报告和混淆矩阵
print("\nOptimized Model Classification Report:\n", classification_report(y_test, y_pred_best))
cm_best = confusion_matrix(y_test, y_pred_best)
print("Optimized Confusion Matrix:\n", cm_best)

# 可视化优化后模型的混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="PuBu", cbar=True)
plt.title("Optimized Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.tight_layout()
plt.show()

# 7. 模型的重要性分析（Feature Importance）
# 随机森林模型可以计算各个特征的重要性，下面绘制特征重要性图，帮助理解哪些特征对分类任务贡献最大
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
plt.title("Feature Importance", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.show()
#创建一个新的图形窗口，设置画布尺寸为宽12英寸、高8英寸。确保图表足够大，避免特征名称重叠，提升可读性。
#X轴数据为排序后的特征重要性值（数值型）；Y轴数据为对应的特征名称（标签型）；使用蓝-绿-黄渐变配色方案，美观且色盲友好
#设置图表标题为“Feature Importance”，字体大小16。 明确图表主题，标题通常加粗且略大于其他文本。
#设置X轴标签为“Importance”，字体大小14。 说明X轴度量的是特征重要性数值（如基尼重要性或SHAP值）。
#设置Y轴标签为“Feature”，字体大小14。 标明Y轴条目为特征名称（如“年龄”“收入”等）。
#自动调整子图间距，避免文字截断或重叠。 在特征数量较多时尤其重要，确保图表元素完整显示。
#渲染并显示图形。 在Jupyter等交互环境中需此命令才会输出图表。 


# 8. ROC 曲线与 AUC 分数评估
# 计算 ROC 曲线，并绘制 ROC 曲线图，评估模型分类效果
y_proba = best_rf.predict_proba(X_test)[:, 1]  # 获取正例概率
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("\nOptimized Model ROC AUC: {:.4f}".format(roc_auc))
#作用：获取测试集中每个样本属于正类（标签为1）的概率
#predict_proba() 返回一个二维数组，每行对应一个样本，列表示属于类别0和1的概率
#[:, 1] 提取所有行（样本）的第二列，即预测为正类的概率
#意义：ROC曲线的绘制需要基于概率值，而非直接预测的类别标签
#输入：y_test：测试集的真实标签（0或1）；y_proba：模型预测的正类概率
#输出：fpr（False Positive Rate）：假阳性率（误判负类为正类的比例）；tpr（True Positive Rate）：真阳性率（正确识别正类的比例，即召回率）；thresholds：生成不同FPR/TPR时使用的概率阈值（从高到低排序）
#原理：通过动态调整分类阈值（默认0.5），计算不同阈值下的FPR和TPR

# 可视化 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Receiver Operating Characteristic", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
#ROC曲线（橙色实线）：横轴（FPR）：负类样本被错误分类的比例（越小越好）；纵轴（TPR）：正类样本被正确分类的比例（越大越好）
#曲线越靠近左上角，模型性能越好；对角线（蓝色虚线）；表示随机猜测模型的性能（AUC=0.5），作为基准线；AUC值：显示在图例中（例如area = 0.9831），直接量化模型性能
#模型性能优秀：AUC值为0.9831，远高于随机猜测（0.5），接近完美分类（1.0）；说明模型能够有效区分正负类样本
#ROC曲线的实际意义：如果希望减少假阳性（如医疗诊断中避免误诊），可选择高阈值；如果希望捕获更多正类（如垃圾邮件检测），可选择低阈值
#适用场景：当数据存在类别不平衡时，AUC比准确率（Accuracy）更能反映模型性能

# 9. 交叉验证与模型稳定性评估
# 采用交叉验证进一步评估模型的稳定性和鲁棒性
cv_scores = cross_val_score(best_rf, X_scaled, y, cv=10, scoring='accuracy')
print("\nCross-validation Accuracy Scores:\n", cv_scores)
print("Mean CV Accuracy: {:.4f}".format(np.mean(cv_scores)))
#作用：使用10折交叉验证评估模型的泛化性能；best_rf：已通过调参优化的随机森林模型；X_scaled：标准化后的特征数据；Y：目标变量（标签）；cv=10：将数据分为10个子集，依次用其中9个训练、1个测试；scoring='accuracy'：以分类准确率作为评估指标

# 可视化交叉验证结果
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, marker='o', linestyle='--', color='teal')
plt.title("Cross-validation Accuracy Scores", fontsize=16)
plt.xlabel("Fold Number", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.xticks(range(1, 11))
plt.ylim(0.90, 1.00)
plt.grid(True)
plt.tight_layout()
plt.show()
#可视化目的：观察模型在不同数据子集上的稳定性；发现潜在问题（如第2折准确率显著下降）
#图形特征：横轴为折数（1到10），纵轴为准确率（范围限制在0.9~1.0）；折线图 + 圆形标记 + 虚线，颜色为青色；网格线辅助观察波动
#作用：创建新的图形窗口；参数：figsize=(10, 6) 设置图形尺寸为宽10英寸、高6英寸；目的：确保图表有足够空间清晰展示数据
#核心绘图函数：创建折线图。range(1, 11)：X轴数据，表示第1到第10折交叉验证；cv_scores：Y轴数据，包含每折验证的准确率分数列表；marker='o'：每个数据点用圆圈标记；linestyle='--'：使用虚线连接数据点；color='teal'：线条和标记使用蓝绿色
#作用：设置图表标题；参数：标题文本和字体大小16（比轴标签稍大）
#作用：设置X轴标签；说明：标注X轴表示交叉验证的折数
#作用：设置Y轴标签；说明：标注Y轴表示模型准确率（0-1范围）
#作用：显式设置X轴刻度：目的：确保X轴精确显示1-10的整数刻度（每折一个）
#作用：设置Y轴显示范围；参数：从0.90到1.00（90%-100%准确率范围）；目的：放大关键区域，使准确率差异更明显
#作用：启用网格线；目的：便于读取数据点的精确值，提高可读性
#作用：自动调整布局；目的：防止标签被截断，优化图表元素间距
#作用：渲染并显示图形

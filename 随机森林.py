本研究旨在基于随机森林算法构建环境驱动因子解析模型，重点探讨土壤pH值、温度梯度、有机碳含量及含水率等关键参数对反硝化作用动力学特征（以潜在反硝化速率为表征）以及氧化亚氮（N2O）排放通量的影响机制，对相关驱动因子的feature important进行分析
鉴于原始农业环境数据存在异质性较高、信噪比不足等局限性，导致非线性模型的预测效能未能显著优于传统线性回归方法，本研究转而采用scikit-learn机器学习库中的乳腺癌数据集进行方法学验证。该替代数据集可有效模拟多变量交互作用下的特征解析过程，后续将系统阐述各算法组件的功能实现及其生态学意义解析方法。
郑鑫、刘英杰、黄梦甜、祝婧怡构造运营环境，下载nump包、pandas包时用到了Anconda prompt
郑鑫进行代码复刻时用到了Vs code
祝婧怡、黄梦甜、刘英杰在编码实现过程中，通过人工智能驱动的方法论，借助AI软件如DeepSeek、python等对多维数据进行建模与特征解析，并采用特征重要性量化分析框架实现驱动因子的动态权重评估。
# 库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
导入 NumPy 库（用于科学计算），并简写为 np以提供高效的数组（ndarray）操作、数学函数、线性代数等
导入 Pandas 库（用于数据分析），并简写为 pd以提供 DataFrame（类似Excel表格的数据结构），支持数据清洗、统计分析等
导入 Matplotlib 的绘图模块，并简写为 plt以基础绘图库，可生成折线图、散点图、直方图等
导入 Seaborn 库（基于Matplotlib），并简写为 sns以提供更美观的统计图表（如热力图、箱线图），简化复杂可视化

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
这段代码导入了 scikit-learn（sklearn） 库中的多个模块和函数，主要用于 机器学习建模、数据预处理、模型评估 等任务。以下是逐行解释：
从 sklearn.datasets 加载 乳腺癌数据集（Breast Cancer Wisconsin Dataset），数据集内容包含肿瘤特征（如半径、纹理等）和对应的标签（良性 0 / 恶性 1）
train_test_split：将数据集随机划分为训练集 和测试集，GridSearchCV：通过网格搜索进行超参数调优（穷举所有参数组合，选择最佳模型），cross_val_score：使用 交叉验证 评估模型性能（如准确率）
导入随机森林算法
confusion_matrix：生成混淆矩阵（显示真阳性、假阳性等分类结果），classification_report：输出分类报告（精确度、召回率、F1值等），roc_curve 和 auc：绘制ROC曲线并计算曲线下面积（AUC），评估模型区分能力，roc_auc_score：直接计算AUC得分（值越接近1，模型越好）。
对数据进行 标准化（均值为0，方差为1），提升模型稳定性


# 1. 数据加载与预处理
# 载入乳腺癌数据集，该数据集包含 569 个样本，每个样本有 30 个特征和二分类标签（良性/恶性）
# 其中load_breast_cancer()是加载乳腺癌数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
这段代码主要用于加载和预处理Scikit-learn内置的乳腺癌数据集，以下是逐行解释：
加载Scikit-learn内置的威斯康星乳腺癌诊断数据集
将特征数据转换为Pandas DataFrame
将目标变量转换为Pandas Series

# 输出数据集基本信息
# 相应代码如下：
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
这段代码用于绘制乳腺癌数据集中各特征之间的相关性热力图，以下是逐行解释：
创建绘图画布:设置图形大小为14英寸(宽)×10英寸(高),为后续绘图准备足够大的画布空间
计算特征相关系数矩阵:使用Pearson相关系数（默认）,取值范围：[-1, 1],1表示完全正相关，-1表示完全负相关,输出结构：30×30的对称矩阵（对应30个特征）
绘制热力图:annot=True 作用为显示数值，取值默认False；fmt=".2f" 作用为数值格式，取值保留两位小数；cmap="RdBu_r" 作用为颜色映射，取值红-蓝渐变（_r表示反转）
第4/5/6行用来添加图表修饰，设置标题字号16，坐标轴标签字号14，实际分析时可替换为更有意义的轴标签
第7/8行用来布局优化，tight_layout用来自动调整子图参数避免标签重叠，show用来显示图形（在Jupyter等环境中可省略）

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
#random_state=42: 设置随机种子为42。随机森林算法在训练过程中会使用随机性，通过设置 random_state 参数，可以确保每次运行代码时生成的随机数相同，从而保证结果的可重复性。
rf.fit(X_train, y_train)
#rf.fit(X_train, y_train)的作用是使用训练数据 X_train 和 y_train 来训练随机森林模型。训练过程中，随机森林会构建多棵决策树，并通过集成这些树的预测结果来提高模型的准确性和稳定性
y_pred = rf.predict(X_test)
#y_pred = rf.predict(X_test)的作用是使用训练好的随机森林模型 rf 对测试数据 X_test 进行预测，并将预测结果存储在变量 y_pred 中。

# 输出分类报告和混淆矩阵
print("\nInitial Model Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
#这段代码用于评估一个初始模型（在这里是随机森林模型）的分类性能，并打印分类报告和混淆矩阵。

# 可视化混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.tight_layout()
#plt.tight_layout: 自动调整子图参数，以确保子图之间有足够的空间，避免标签或标题重叠。这在绘制复杂图形时非常有用。
plt.show()
#这段代码使用 matplotlib 和 seaborn 库来可视化混淆矩阵（cm），使其更直观易读。

# 5. 随机森林超参数调优
# 通过 GridSearchCV 网格搜索方法寻找最佳超参数组合
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
#这段代码定义了一个参数网格，用于在超参数调优过程中尝试不同的超参数组合，以找到最佳的模型性能。n_estimators 和 max_depth 的组合可以控制模型的复杂度。min_samples_split 和 min_samples_leaf 的组合可以控制树的分裂严格程度。bootstrap 决定了树的构建方式。这些参数的组合会影响模型的性能和计算成本。通过工具可以自动尝试所有可能的组合，找到最优的超参数。

# 采用 5 折交叉验证
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                           scoring='accuracy')
#使用 GridSearchCV 对 RandomForestClassifier 模型进行超参数调优，遍历 param_grid 中的所有超参数组合，使用 5 折交叉验证评估每组超参数的性能，并行计算以加速调优过程，输出日志信息以便跟踪进度。最终选择使准确率（accuracy）最高的超参数组合。
# 训练调优
grid_search.fit(X_train, y_train)
#是启动超参数调优和模型训练的核心代码。它通过交叉验证和穷举搜索找到最佳超参数组合，并训练出最终的模型。通过这种方式，可以显著提升模型的性能和泛化能力。

# 输出最佳超参数及最佳得分
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Accuracy: {:.4f}".format(grid_search.best_score_))

# 6. 使用最优参数构建优化后的随机森林模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)
#利用网格搜索找到最佳的随机森林模型，并使用该模型对测试数据进行预测。通过合理地设置参数网格和评估指标，可以有效地优化模型的性能。

# 输出优化后模型的分类报告和混淆矩阵
print("\nOptimized Model Classification Report:\n", classification_report(y_test, y_pred_best))
cm_best = confusion_matrix(y_test, y_pred_best)
print("Optimized Confusion Matrix:\n", cm_best)
#这段代码用于在分类任务中评估机器学习模型的性能。它输出了分类报告（classification report）和混淆矩阵（confusion matrix），这些是常用的评估指标，用于了解模型在测试数据上的表现。

# 可视化优化后模型的混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="PuBu", cbar=True)
plt.title("Optimized Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.tight_layout()
plt.show()
#使用了 matplotlib 和 seaborn 库来可视化混淆矩阵（cm_best），使其更直观易读。plt.tight_layout: 自动调整子图参数，以确保子图之间有足够的空间，避免标签或标题重叠。这在绘制复杂图形时非常有用。

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

# 8. ROC 曲线与 AUC 分数评估
# 计算 ROC 曲线，并绘制 ROC 曲线图，评估模型分类效果
y_proba = best_rf.predict_proba(X_test)[:, 1]  # 获取正例概率
#作用：获取测试集中每个样本属于正类（标签为1）的概率
#predict_proba() 返回一个二维数组，每行对应一个样本，列表示属于类别0和1的概率
#[:, 1] 提取所有行（样本）的第二列，即预测为正类的概率
#意义：ROC曲线的绘制需要基于概率值，而非直接预测的类别标签
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
#输入：y_test：测试集的真实标签（0或1）；y_proba：模型预测的正类概率
#输出：fpr（False Positive Rate）：假阳性率（误判负类为正类的比例）；tpr（True Positive Rate）：真阳性率（正确识别正类的比例，即召回率）；thresholds：生成不同FPR/TPR时使用的概率阈值（从高到低排序）
#原理：通过动态调整分类阈值（默认0.5），计算不同阈值下的FPR和TPR
roc_auc = auc(fpr, tpr)
print("\nOptimized Model ROC AUC: {:.4f}".format(roc_auc))

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


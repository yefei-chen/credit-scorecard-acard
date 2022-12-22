# 基于评分卡的贷前用户信用风险预测

## 问题概述

评分卡，一般是指用于小贷用户质量评分的评分卡表，它是以分数形式衡量用户信用风险大小的手段，衡量贷款人可能给金融机构带来经济损失的可能性。一般来说，评分卡分数越高，用户信用越好，风险越小。对于个人用户来说，评分卡分为四类：A卡、B卡、C卡和F卡。A卡（Application Scorecard）被称为申请评分卡或者申请者评级模型，用来帮助金融机构来判断是否应该借钱给一个新用户。如果某一个人风险太高，我们就可以拒绝贷款。B卡(Behavior Scorecard）和C卡（Collection Scorecard）分别用于评估借贷过程中用户行为和逾期的风险，F卡（Fraud Scorecard)则用于申请阶段对欺诈用户的识别。

评分卡的原理就是利用历史客户数据，分析出有效的特征，并将连续特征离散化（即分箱）。然后使用逻辑回归模型确定每个特征的权重和基础阈值，最后根据阈值，权重确定基础分数和每个分箱的分数。本文中基于A卡建立逻辑回归模型进行贷前用户的信用风险预测，以帮助金融机构评估是否要批准贷款。

通过评分卡进行贷前用户信用风险评估，不仅可以减少人工审核量，也可以提高放贷质量，是中小型金融机构必备的模型。


## 项目目标

本文以Kaggle上的借贷数据（数据来源：<a href="https://www.kaggle.com/competitions/GiveMeSomeCredit/data">Kaggle - Give Me Some Credit</a>）为例，分别进行数据清洗、EDA（探索性数据分析）、特征工程（包括数据分箱和特征选取）、逻辑回归建模、评分卡转换五个步骤，构建完整的评分卡模型，并生成可供生产部署的分组逻辑和评分阈值。

## 理解数据

Give Me Some Credit数据包含四个部分：cs-training.csv、cs-test.csv、sampleEntry.csv、Data Dictionary.xls。其中cs-training.csv为训练集，包含10个属性特征和1个结果特征SeriousDlqin2yrs，表示贷款人逾期与否。其余特征的含义和数据类型在Data Dictionary.xls作出具体解释，如下表所示。cs-test.csv和sampleEntry.csv为提交用的测试数据集。

在本文将仅使用cs-training.csv构建并验证模型效果。该训练集包括15w条借贷数据。

<table cellspacing="0" style="border-collapse:collapse; height:233px; width:794px">
	<tbody>
		<tr>
			<td style="background-color:#c0c0c0; border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><strong><span style="font-family:Arial">Variable&nbsp;Name</span></strong></span></span></td>
			<td style="background-color:#c0c0c0; border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><span style="font-size:13px"><span style="color:#000000"><strong><span style="font-family:Arial">Description</span></strong></span></span></td>
			<td style="background-color:#c0c0c0; border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><strong><span style="font-family:Arial">Type</span></strong></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><strong><span style="font-family:Arial">SeriousDlqin2yrs</span></strong></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><strong>逾期90天或更久</strong></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><strong><span style="font-family:Arial">Y/N</span></strong></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:24px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">RevolvingUtilizationOfUnsecuredLines</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:24px; text-align:center; vertical-align:top; white-space:normal; width:442px"><span style="color:#000000; font-family:Arial">信用卡和个人信用欠款总额（房产除外）和非分期债务（如车贷）之和除以信用额度之和</span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:24px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">percentage</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">age</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><span style="color:#000000; font-family:Arial">借款人年龄</span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">NumberOfTime30-59DaysPastDueNotWorse</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; vertical-align:bottom; width:442px">
			<p style="text-align:center"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">借款人逾期30-59天（但在过去两年中没有更长的逾期时长）的次数</span></span></span></p>
			</td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">DebtRatio</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; vertical-align:bottom; width:442px">
			<p style="text-align:center"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">每月债务支付、赡养费、生活费用之和除以每月总收入</span></span></span></p>
			</td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">percentage</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">MonthlyIncome</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><span style="color:#000000; font-family:Arial">每月收入</span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">real</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">NumberOfOpenCreditLinesAndLoans</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; vertical-align:bottom; width:442px">
			<p style="text-align:center"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">未结贷款数量（分期付款例如车贷或抵押贷款）和信用额度（如信用卡）</span></span></span></p>
			</td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">NumberOfTimes90DaysLate</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><span style="color:#000000; font-family:Arial">借款人逾期90天或更久的次数</span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">NumberRealEstateLoansOrLines</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; vertical-align:bottom; width:442px">
			<p style="text-align:center"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">抵押贷款和房产贷款的数量（包括房屋净值信贷额度）</span></span></span></p>
			</td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">NumberOfTime60-89DaysPastDueNotWorse</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">借款人逾期60-89天（但在过去两年中没有更长的逾期时长）的次数</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
		<tr>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:237px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">NumberOfDependents</span></span></span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:442px"><span style="color:#000000; font-family:Arial">家庭中的受抚养人数（例如配偶、子女等），不包括他们自己</span></td>
			<td style="border-color:#000000; border-style:solid; border-width:0.7px; height:19px; text-align:center; vertical-align:bottom; width:112px"><span style="font-size:13px"><span style="color:#000000"><span style="font-family:Arial">integer</span></span></span></td>
		</tr>
	</tbody>
</table>


## 数据清洗
详见notebook

## 探索性数据分析
所有数据中，6%的用户发生违约，而未违约用户的占比为93.3%。可以发现，这是一个非常不平衡的数据集，会影响预测的结果。因此，在建模之前，需要对样本进行平衡。

<p>RevolvingUtilizationOfUnsecuredLines：</p>

<ul>
	<li>RevolvingUtilizationOfUnsecuredLines，即信用卡和个人信用欠款总额（房产除外）和非分期债务（如车贷）之和除以信用额度之和，根据经验判断，随着该值增加，违约人数比例也会上升</li>
	<li>该特征的分布是右偏的，最大值为50708，说明很多用户欠款已经超过信用额度，甚至超过5w多倍</li>
</ul>

<p>Age：</p>

<ul>
	<li>Age，即贷款人年龄，分布基本符合正态分布规律</li>
	<li>排除10岁以下和100岁以上的异常用户</li>
	<li>可以看出未违约人群远大于违约人群，且违约人群趋向于年轻化</li>
</ul>

<p>NumberOfTime30-59DaysPastDueNotWorse/NumberOfTime60-89DaysPastDueNotWorse/NumberOfTimes90DaysLate：</p>

<ul>
	<li>这三个指标用于衡量用户逾期分别为30-59、60-89、90天以上的次数</li>
	<li>逾期的次数通常不超过20次</li>
	<li>逾期次数越多，显然违约人数的比例会上升</li>
</ul>

<p>Dept Ratio：</p>

<ul>
	<li>Dept Ratio，即每月债务支付、赡养费、生活费用之和除以每月总收入（负债率），根据经验判断，随着该值增加，违约人数比例也会上升</li>
	<li>该特征的分布是右偏的，和RevolvingUtilizationOfUnsecuredLines类似，最大值为329664，说明很多用户负债已经已经远远超过收入，甚至超过32w多倍</li>
</ul>


<p>MonthlyIncome：</p>

<ul>
	<li>MonthlyIncome，即每月收入，平均值为6040，根据经验，收入越高，违约人群的比例应越低</li>
	<li>该特征基本符合正态分布，同时收入的分布体现了二八定律：80%的财富集中在20%的手中</li>
</ul>

<p>NumberOfOpenCreditLinesAndLoans/NumberRealEstateLoansOrLines：</p>

<ul>
	<li>NumberOfOpenCreditLinesAndLoans和NumberRealEstateLoansOrLines衡量的是未结贷款数量，通常数量越多，违约人群的比例越大</li>
	<li>NumberOfOpenCreditLinesAndLoans的数目平均要大于NumberRealEstateLoansOrLines，这可能与贷款规模有关，小额贷款的持有比例更多</li>
	<li>NumberOfOpenCreditLinesAndLoans基本呈正态分布，可以看出目前小额贷款或信用卡消费越来越普及</li>
	<li>NumberRealEstateLoansOrLines分布不太均匀，房贷持有比例还是较低的</li>
</ul>

<p>NumberOfDependents：</p>

<ul>
	<li>NumberOfDependents，即家庭人数，根据常识推断，家庭人数越多，违约比例越低</li>
	<li>家庭人数也是呈右偏分布的，最大家庭人数有20人，然而占据最大群体的是独身人群</li>
</ul>

## 特征工程
在这一部分，将会对样本数据进行分箱，并通过IV值选取典型特征，再结合badrate法查看并调整分箱。

### 数据分箱
评分卡最重要的步骤就是数据分箱。分箱的本质就是将连续特征离散化，好让拥有不同属性的用户被分成不同的类别，并且可以避免连续型特征的取值过于稀疏而影响模型学习效果。

分箱最传统的方法是badrate分析法，它的原理是先对x进行分组，然后分析每组x的badrate（违约人群的比例）是否有固定趋势来确定y是否与x相关，并通过调整分组逻辑，使x分组后与y的关系更加稳定。通常分箱后，badrate应随着箱的取值大小而单调变化。badrate的分组逻辑必须具有强业务性，因此通常训练后的模型更贴近业务，但是需要大量的人力和时间。

另一种方法就是自动分箱，又分为无监督分箱和有监督分箱。无监督分箱中最常用的是等频分箱和等距分箱，分别是令每箱的样本数相等和每箱的宽度相等。有监督分箱常用的有卡方分箱、决策树分箱、Best-KS分箱。

分箱的数目，不宜太多，最好不超过10个，也不宜太少，4～5个最佳，因为箱子越少，损失的信息越多，银行业通过IV来衡量特征的信息量以及特征对预测函数的贡献，详情将在下一节介绍

在实际运用中，常常将传统和自动的方法相结合。例如通过自动分箱和IV值筛选特征，并查看所有特征的badrate，筛选掉完全经不住业务考验的特征。

本文将基于决策树算法对数据分箱，具体步骤如下：

<ol>
	<li>利用sklearn决策树，DecisionTreeClassifier的.tree_属性获得决策树的节点划分值</li>
	<li>基于上述得到的划分值，利用pandas.cut函数对特征进行分箱</li>
	<li>计算各个分箱的WOE、IV值以及badrate趋势</li>
</ol>

### 特征选取

将特征分箱的边界确认后，就可以计算IV值以筛选特征。在了解IV值之前，首先了解WOE的含义。

WOE的全称是Weight of Evidence，即证据权重，它是银行业用来衡量违约概率的指标，通常要在特征分箱处理后才能进行WOE编码，对于特征X第i箱的WOE值，其公式如下：

$$WOE_{i}=ln\frac{Bad_{X=X_i}/Bad_{total}}{Good_{X=X_i}/Good_{total}}=ln\frac{Bad_{X=X_i}/Good_{X=X_i}}{Bad_{total}/Good_{total}}$$

其中$Bad_{total}$代表坏样本总数（即违约样本），$Good_{total}$代表好样本总数，$Bad_{X=X_i}$代表第i箱的坏样本总数，$Good_{X=X_i}$代表第i箱的好样本总数。按照特征WOE映射表，就可以把特征的组别映射成对应的WOE值。

可以发现，WOE表示的实际上是“当前分箱中坏客户和好客户比值”和“所有样本中坏客户和好客户比值”的差异，显然，WOE越大，这种差异越大，坏客户的可能性越大，该分箱内的样本违约的可能性越大。

有了WOE的基本概念，接下来进一步了解IV。IV的全称是Information Value，即信息价值。通常在用逻辑回归模型时，需要对自变量进行筛选，即挑选出“入模特征”，需要考虑的一个最主要因素就是“特征的预测能力”。IV就是这样一种指标，类似的指标还有信息增益、基尼系数等。和WOE一样，每个分箱i会有一个对应的IV值，其公式如下：

$$IV_i=(\frac{Bad_{X=X_i}}{Bad_{total}}-\frac{Good_{X=X_i}}{Good_{total}})*ln\frac{Bad_{X=X_i}/Bad_{total}}{Good_{X=X_i}/Good_{total}}=(\frac{Bad_{X=X_i}}{Bad_{total}}-\frac{Good_{X=X_i}}{Good_{total}})*WOE_i$$

将特征各分箱的IV值相加，就可以计算整个特征的IV值：

$$IV=\sum_{i}^nIV_i$$

信息论中，评估两个分布的距离（差异）可以用KL散度，又称为相对熵，对于分布P(x)和分布Q(x)，其Q对P的散度为：

$$D_{KL}(P||Q)=\sum_{x\in{X}}P(x)*ln\frac{P(x)}{Q(x)}$$

事实上，IV的意义就是违约用户（坏客户）与未违约客户（好客户）的距离/散度，令$G=\frac{Good_{X=X_i}}{Good_{total}}$代表每个分箱好客户的分布，$B=\frac{Bad_{X=X_i}}{Bad_{total}}$ 代表每个分箱坏客户的分布，则好客户相对坏客户的KL散度为：

$$D_{KL}(B||G)=\sum_{i}\frac{Bad_{X=X_i}}{Bad_{total}}*ln\frac{Bad_{X=X_i}/Bad_{total}}{Good_{X=X_i}/Good_{total}}$$

坏客户相对好客户的KL散度为：

$$D_{KL}(G||B)=\sum_{i}\frac{Good_{X=X_i}}{Good_{total}}*ln\frac{Good_{X=X_i}/Good_{total}}{Bad_{X=X_i}/Bad_{total}}$$

二者之和即信息量IV：

$$IV=D_{KL}(B||G)+D_{KL}(G||B)$$

IV值越高，说明分布的区分度越高，也就是特征X对区分目标y的作用越大，即特征的价值越高。通常：

<ul>
	<li>IV&lt;0.02：几乎没有区分度</li>
	<li>0.02&lt;=IV&lt;0.1：有微弱的区分度</li>
	<li>0.1&lt;=IV&lt;0.3：有明显的区分度</li>
	<li>IV&gt;=0.3：较强区分度</li>
</ul>

### WOE转换
通过特征分箱，将原数据映射到不同的区间，但如果直接把箱序号作为特征变量值是不够理想的，因为组号是等距的，而badrate不等距，WOE可以把相对于bad rate显现非特性的特征转换为线性的，这对于逻辑回归非常有必要。除此之外，转成WOE的数据也更加稳健和容易操作。



## 逻辑回归
基于上一节转换得到的WOE数据，进行数据归一化、数据集划分、数据平衡、模型训练与模型评估，从而构建用户信用评分的逻辑回归模型。逻辑回归是一种用于二分类的模型，它输出属于类别1的概率，其表达式如下：

$$P(x)=Sigmoid(XW)=\frac{1}{1+e^{-(w1x1+w2x2+....+w_kx_k+b)}}$$

其中，b是模型的截距，w1-wk为模型的系数。可以看出，逻辑回归模型其实相当于用线性函数综合所有变量，再用sigmoid函数将综合值转换为概率值。逻辑回归用概率最大化作为损失函数，即取何值时，模型预测正确的概率最大。其损失函数如下：

$$L(W)=\sum_{i=1}^{m}[y_i*ln(P_i)+(1-y_i)*ln(1-P_i)]$$

逻辑回归模型无法求得精确解，通常使用梯度下降等算法进行数值求解，损失函数的梯度公式为：

$$\frac{∂L(W)}{∂W}=X^T(p-y)$$

### 数据归一化

### 数据集划分

### 数据集平衡
对于不平衡数据，通常采用欠采样或是过采样的方式平衡数据。欠采样即从大数目类别样本选取和小数目类别样本数目相当的样本，然后和少数目类别样本组成新的数据集，使得在新的数据集中正负样本比例相当。过采样即少数类中一个样本抽取多次，从而使正负样本数目接近，再进行学习。在这里使用SMOTE算法对训练数据进行过采样。

### 模型训练

<img width="622" alt="image" src="https://user-images.githubusercontent.com/49276153/209112845-b28c7819-3462-4542-8b03-ecaca53fd7eb.png">

通常AUC和投产的关系如下：
<ul>
	<li>AUC&gt;0.63：模型对y有区分度（不可投产）</li>
	<li>AUC&gt;0.68：有效益（不可投产）</li>
	<li>AUC&gt;0.73：模型可投产</li>
</ul>

由ROC曲线可知，模型的AUC达到0.85，整体性能较好，基本可以满足需求。在实际生产工作中，应花费更多的时间和业务人员调整AUC曲线以达到最优模型。

## 评分卡转换

逻辑回归模型得到的是用户违约的概率，现需要将其通过评分卡转换为用户评分，转换方法是将逻辑回归模型的线性部分抽出，并作一定的线性缩放。转换步骤如下。

1. 令
$$odds=\frac{P}{1-P}$$
它是坏用户的概率与好用户的概率的比值。根据逻辑回归原理，
$$P=\frac{1}{1+e^{-(wx+b)}}$$
变化可得，
$$ln(\frac{P}{1-P})=wx+b$$
或
$$ln(odds)=wx+b$$


2. 评分卡背后的逻辑是odds的变动与评分变动的映射，设计公式如下：

$$Score=A-B*ln(odds)$$

其中A与B是常数，B前面取符号是为了满足“违约概率越低，得分越高”，即“高分高信用低风险”。


3. 定义两个假设:

    a. 基准分。当odds为初始值θ0时的分数P0。通常业界的风控策略基准分设置在600左右，其公式为：
    
    $$P_0=A-B*ln(θ_0)$$
    
    b. PDO(point of double)。即比率翻番时分数的变动值。例如假设当odds翻倍时，分值减少20，则PDO=30
    
    
4. 根据θ0，P0和PDO计算A和B：

    a. 首先根据PDO的定义可知：
    
    $$P_0-PDO=A-B*ln(2θ_0)$$
    
    b. 和3a联立可解得：
    
    $$A=P_0+B*ln(θ_0)$$
    
    $$B=\frac{PDO}{ln2}$$
    

5. 将A，B的值代入1和2，把分箱映射为分数：

$$Score=A-B\{b+w_1x_1+...+w_nx_n\}$$


其中b是逻辑回归模型的截距，w是逻辑回归模型的参数，变量x是出现在最终模型的入模变量，且所有的入模变量都进行了WOE编码。将上式进一步转化可得：


$$Score=A-B*b-B\{w_1x_1+...+w_nx_n\}$$


其中A-B*b又称为基础分BaseScore，-B*wnxn为每个特征每个分箱的特征分FeatureScore。在计算用户得分时，只需算得用户各个特征所属分组，然后在表中查得特征得分，再加上基础分，就是用户的总评分。

<img width="488" alt="image" src="https://user-images.githubusercontent.com/49276153/209113603-ba0f7cb6-f738-467c-8903-1d6cdbb6c94d.png">


## 评分阈值表

生成了评分卡，通过评分卡就可以对新进的用户进行评分，从而评估是否发放贷款。在模型投产时，需要设定评分阈值，将低于评分阈值的用户拒绝。评分阈值的设定需要借助评分阈值表，并结合业务而设定。

阈值表由建模样本的评分统计得到，它展示了不同评分阈值给业务带来的效果。阈值表的统计需要先计算出各个样本的评分，再对评分按分段分组，统计每个分段的统计信息。

<img width="1036" alt="image" src="https://user-images.githubusercontent.com/49276153/209114185-bd022c36-c93d-4373-ba30-cf33c752974d.png">

<img width="566" alt="image" src="https://user-images.githubusercontent.com/49276153/209114255-97a16f55-64b8-4205-8fc3-db3c1d19e6f0.png">

## 总结

本文基于kaggle上的用户信用数据构建逻辑回归模型，并转化为评分卡，进行用户是否有违约可能性的预测。模型最终保留的评分特征有RevolvingUtilizationOfUnsecuredLines、age、NumberOfTime30-59DaysPastDueNotWorse、NumberOfTimes90DaysLate、NumberOfTime60-89DaysPastDueNotWorse这五项，模型的测试AUC达到0.85，具有较好的性能。

最终模型概率表达式如下：

$$P=\frac{1}{1+e^{-[0.68056389*RevolvingUtilizationOfUnsecuredLines_{woe}+0.52225767*age_{woe}+0.65002843*NumberOfTime30-59DaysPastDueNotWorse_{woe}+0.64861883*NumberOfTimes90DaysLate_{woe}+0.49774068*NumberOfTime60-89DaysPastDueNotWorse_{woe}+0.02552065319436435
]}}$$

分数表达式如下：

$$Score=712.8771237954945
-28.85390081777927*[0.68056389*RevolvingUtilizationOfUnsecuredLines_{woe}+0.52225767*age_{woe}+0.65002843*NumberOfTime30-59DaysPastDueNotWorse_{woe}+0.64861883*NumberOfTimes90DaysLate_{woe}+0.49774068*NumberOfTime60-89DaysPastDueNotWorse_{woe}+0.02552065319436435
]$$


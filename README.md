# 机器学习前沿项目二修正

[TOC]

# 对于Prompt Learning的理解

原始的数据相当于一个医术高明的医生，prompt相当于医生的一个动作组，Large Language Model充当一个翻译机，而分类任务则是一个无法处理小孩哭闹的愚蠢的大人，最终得到的结果就是小孩的状态。医生与大人语言不通，直接沟通效果会很差

在更新过程中，data就是医生给出的解决方案，prompt是医生为了能够让Large Language Model理解意思所作出的有益于理解的肢体语言，Large Language Model则给出更适合分类任务的数据分布，帮助分类任务进行参数更新

# 第一次尝试

尝试soft Prompt的方法

完整组件分为：PromptModel–》Large Language Model –》MaskModel（基本结构)

## preProcessing

将得到的数据进行标准化（预防孤立数据和异常点），将其进行等长离散化，将标准化后的数据最大最小值之间进行区间划分，用区间号代替原本的数据（如果直接塞浮点数数据的话感觉没有容错率，所以离散化给数据一点柔性）

## promptModel

==(最好画个图)==

### promptGenerate

类内部生成一个随机数buffer，buffer经过gru和mlp，得到prompt_length长度的一组浮点数，然后将这个浮点数映射到tokenizer的词表空间里取，

### mask_slice

类内部生成一个随机数buffer，经过mlp得到2长度的一个浮点数，将其映射到prompt常数的整数空间，然后分别作为mask和slice

### 分割prompt

首先将得到的prompt插入mask，然后根据slice将其区分为 beforePrompt和afterPrompt分别表示在data前面的prompt和在data后面的prompt，返回beforePrompt，afterPrompt和mask的位置(在promptGenerate整数映射时无法保证得到的原始prompt不包含mask标签，所以需要记录)

## Large Language Model

使用的Large Language Model：bert-large-uncased

### 过程

将torch.cat((beforePrompt,data,afterPrompt))拼接，传入Large Language Model，得到bert的last_hidden_state,取出前面得到的maskpos，将maskpos处的数据取出(与prompt Learning的预测对应的词不一样，因为我不知道bert应该怎样把输出还原回词的概率)

## MaskModel

将上面得到的数据，将其放入mlp中进行多分类

## train process

promptModel和maskModel更新，Large Language Model进行冻结，进行多分类，使用交叉熵损失函数，Optimizer使用Nadam

## Fault

在刚开始进行检测的时候，因为在二维数据上有较好的表现（优于原来的mlp），没有检查prompt打印的文本，没有注意到loss的反向传播会被Large Language Model的Embedding层挡住，promptModel完全无法更新，在打印了prompt的文本后才发现prompt根本没变，继而得出结论，模型设计有问题

## Concolusion

经验总结，promptModel-》llm-》maskModel的三层结构贯彻始终

从last_hidden_state中取出mask——pos的数据进行mlp

# 第二次尝试

## target

希望能够更新promptModel的参数

## 方案

在旁边建立一条连接，通过cutModel将torch.cat(prompt,mask_slice)进行mlp得到与maskpos中得到的数据同样大小的数据，进行mseloss，继而使得loss能够更新

## 效果

promptModel可以更新，但是效果非常拉跨（比第一次差）

## 猜测

LLM生成的更好的数据分布还没有被mlp学习到，数据分布就已经变了，继而导致始终无法收敛

## concolusion

无法通过神经网络的方式更新参数

### Final Version

由上一个尝试得知，无法通过神经网络的方式更新参数，所以放弃了第一次尝试中使用gru+mlp，mlp分别生成prompt和mask_slice

### promptModel

将原来的参数获取改为直接从正态分布中随机，不需要神经网络进行参数的更新

## 训练过程

promptModel首先生成一个prompt和mask，slice，然后进行训练，当valid_loss<best_valid_loss时，保存模型，反之，not_changed+1直到达到patience，然后重新取样，再重头开始，单次更新实际相当于一个hard prompt，只不过hard prompt的得出方式是根据正太分布随机的，而不是人手动给出的

# dim=3

对于三维数据有以下考虑：

1. 使用同一种prompt
2. 对于不同的时序，使用不同的prompt
3. 使用AutoEncoder进行数据压缩（AutoEncoder得到的数据有损失，实际测试好像特别容易过拟合，所以放弃了）


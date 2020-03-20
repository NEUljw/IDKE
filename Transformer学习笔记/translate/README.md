# Transformer代码实现

**任务：**

机器翻译（德语->英语）

**数据集：**

torchtext中的Multi30k数据集。Multi30k包含约31000条英语句子及对应的德语、法语译文。我们分出29000条句子作为训练集，1014条句子作为验证集，1000条句子作为测试集。

**注：**

论文'the Attention is All you Need'中positional embedding是一个静态的矩阵，然而现如今Transformer体系结构的模型如BERT使用的是可训练的矩阵。因此，这里我们也使用可训练的矩阵作为positional embedding。

**文件说明：**

data.py：加载Multi30k数据集并生成迭代器（iterator）。

model_def.py：Transformer模型的定义。

train_and_eval.py：模型的训练及保存。

test_and_translate.py：加载训练好的模型，进行测试和句子的翻译。

**文本翻译测试：**

最后，我们用训练好的模型进行测试。

首先在训练集里找一句德语，正确的译文是' a woman with a large purse is walking by a gate. '

模型的输出是' a woman with a large purse walks past a gate. S '（S是句子的结尾符）

然后我们在模型没有训练过的数据上进行测试。

在验证集里找一句德语，正确的译文是' a brown dog is running after the black dog. '

模型的输出是' a brown dog runs after the black dog. S '

在测试集里找一句德语，正确的译文是' a mother and her young song enjoying a beautiful day outside. '

模型的输出是' a mother and her son enjoying a beautiful day outside. S '

可以看出，模型的预测效果还是不错的。
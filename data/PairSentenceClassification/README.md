## 数据集下载
地址一：https://cims.nyu.edu/~sbowman/multinli/ <br>
地址二：公众号回复"数据集"获取百度云下载地址！

## 数据格式

```python
{"annotator_labels": ["entailment", "neutral", "entailment", "neutral", "entailment"], "genre": "oup", "gold_label": "entailment", "pairID": "82890e", "promptID": "82890", "sentence1": " From Home Work to Modern Manufacture", "sentence1_binary_parse": "( From ( ( Home Work ) ( to ( Modern Manufacture ) ) ) )", "sentence1_parse": "(ROOT (PP (IN From) (NP (NP (NNP Home) (NNP Work)) (PP (TO to) (NP (NNP Modern) (NNP Manufacture))))))", "sentence2": "Modern manufacturing has changed over time.", "sentence2_binary_parse": "( ( Modern manufacturing ) ( ( has ( changed ( over time ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (NNP Modern) (NN manufacturing)) (VP (VBZ has) (VP (VBN changed) (PP (IN over) (NP (NN time))))) (. .)))" }
{"annotator_labels": ["entailment", "neutral", "entailment", "neutral", "entailment"], "genre": "oup", "gold_label": "entailment", "pairID": "82890e", "promptID": "82890", "sentence1": " From Home Work to Modern Manufacture", "sentence1_binary_parse": "( From ( ( Home Work ) ( to ( Modern Manufacture ) ) ) )", "sentence1_parse": "(ROOT (PP (IN From) (NP (NP (NNP Home) (NNP Work)) (PP (TO to) (NP (NNP Modern) (NNP Manufacture))))))", "sentence2": "Modern manufacturing has changed over time.", "sentence2_binary_parse": "( ( Modern manufacturing ) ( ( has ( changed ( over time ) ) ) . ) )", "sentence2_parse": "(ROOT (S (NP (NNP Modern) (NN manufacturing)) (VP (VBZ has) (VP (VBN changed) (PP (IN over) (NP (NN time))))) (. .)))"}
```

## 数据预处理
由于该数据集同时也可用于其它任务中，因此除了我们需要的前提和假设两个句子和标签之外，还有每个句子的语法解析结构等等。在这里，下载完成数据后只需要执行项目中的`format.py`脚本即可将原始数据划分成训练集、验证集和测试集。格式化后的数据形式如下所示：

```python
From Home Work to Modern Manufacture_!_Modern manufacturing has changed over time._!_1
They were promptly executed._!_They were executed immediately upon capture._!_2
```
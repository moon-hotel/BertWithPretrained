# BertWithPretrained
本项目是一个基于PyTorch从零实现的BERT模型及相关下游任务示例

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

更多关于Transformer内容的介绍可以参考文章[ This post is all you need（层层剥开Transformer）](https://mp.weixin.qq.com/s/uch_AGcSB8OSAeVu2sme8A) ，近4万余字、50张图、3个实战示例（[翻译](https://github.com/moon-hotel/TransformerTranslation) 、[分类](https://github.com/moon-hotel/TransformerClassification) 、[对联生成](https://github.com/moon-hotel/TransformerCouplet) ），带你一网打尽Transformer！

 ## 模型详细解析
- [x] [1. BERT原理与NSL和MLM](https://www.ylkz.life/deeplearning/p10631450/) <br>
- [x] [2. 从零实现BERT网络模型](https://www.ylkz.life/deeplearning/p10602241/) 　　　[代码](model/BasicBert)
- [x] [3. 基于BERT预训练模型的中文文本分类任务](https://www.ylkz.life/deeplearning/p10979382/) 　　　[代码](Tasks/TaskForSingleSentenceClassification.py)
- [x] [4. 基于BERT预训练模型的英文文本蕴含任务](https://www.ylkz.life/deeplearning/p10407402/) 　　　[代码](model/DownstreamTasks/BertForSentenceClassification.py)
- [ ] [5. 基于BERT预训练模型的英文多选项任务](https://www.ylkz.life) 　　　[代码](model/DownstreamTasks/BertForMultipleChoice.py)
- [ ] [6. 基于BERT预训练模型的英文问答任务](https://www.ylkz.life) 　　　[代码](model/DownstreamTasks/BertForQuestionAnswering.py)
- [ ] [7. 基于NSL和MLM任务从头训练BERT任务](https://www.ylkz.life)


## 工程结构
- `bert_base_chinese`目录中是BERT base中文预训练模型以及配置文件

    模型下载地址：https://huggingface.co/bert-base-chinese/tree/main
- `bert_base_uncased_english`目录中是BERT base英文预训练模型以及配置文件

    模型下载地址：https://huggingface.co/bert-base-uncased/tree/main
    
    注意：`config.json`中需要添加`"pooler_type": "first_token_transform"`这个参数
- `data`目录中是各个下游任务所使用到的数据集
    - `SingleSentenceClassification`是今日头条的15分类中文数据集；
    - `PairSentenceClassification`是MNLI（The Multi-Genre Natural Language Inference Corpus, 多类型自然语言推理数据库）数据集；
    - `MultipeChoice`是SWAG问题选择数据集
    - `SQuAD`是斯坦福大学开源的问答数据集1.1版本
    - `WikiText`是维基百科英文语料用于模型预训练
- `model`目录中是各个模块的实现
    - `BasicBert`中是基础的BERT模型实现模块
        - `MyTransformer.py`是自注意力机制实现部分；
        - `BertEmbedding.py`是Input Embedding实现部分；
        - `BertConfig.py`用于导入开源的`config.json`配置文件；
        - `Bert.py`是BERT模型的实现部分；
    - `DownstreamTasks`目录是下游任务各个模块的实现
        - `BertForSentenceClassification.py`是单标签句子分类的实现部分；
        - `BertForMultipleChoice.py`是问题选择模型的实现部分；
        - `BertForQuestionAnswering.py`是问题回答（text span）模型的实现部分；
        - `BertForNSPAndMLM.py`是BERT模型预训练的两个任务实现部分；
- `Task`目录中是各个具体下游任务的训练和推理实现
    - `TaskForSingleSentenceClassification.py`是单标签单文本分类任务的训练和推理实现，可用于普通的文本分类任务；
    - `TaskForPairSentence.py`是文本对分类任务的训练和推理实现，可用于蕴含任务（例如MNLI数据集）；
    - `TaskForMultipleChoice.py`是问答选择任务的训练和推理实现，可用于问答选择任务（例如SWAG数据集）；
    - `TaskForSQuADQuestionAnswering.py`是问题回答任务的训练和推理实现，可用于问题问答任务（例如SQuAD数据集）；
- `test`目录中是各个模块的测试案例
- `utils`是各个工具类的实现
    - `data_helpers.py`是各个下游任务的数据预处理及数据集构建模块；
    - `log_helper.py`是日志打印模块；

## 环境
Python版本为3.6，其它相关包的版本如下：
```python
torch==1.5.0
torchtext==0.6.0
torchvision==0.6.0
transformers==4.5.1
numpy==1.19.5
pandas==1.1.5
scikit-learn==0.24.0
tqdm==4.61.0
```
## 使用方式
### Step 1. 下载数据 
下载完成各个数据集以及相应的BERT预训练模型（如果为空），并放入对应的目录中.
### Step 2. 运行模型 
进入`Tasks`目录，运行相关模型.
### 2.1 中文文本分类任务
```python
python TaskForSingleSentenceClassification.py
```

运行结果:

```python
-- INFO: Epoch: 0, Batch[0/4186], Train loss :2.862, Train acc: 0.125
-- INFO: Epoch: 0, Batch[10/4186], Train loss :2.084, Train acc: 0.562
-- INFO: Epoch: 0, Batch[20/4186], Train loss :1.136, Train acc: 0.812        
-- INFO: Epoch: 0, Batch[30/4186], Train loss :1.000, Train acc: 0.734
...
-- INFO: Epoch: 0, Batch[4180/4186], Train loss :0.418, Train acc: 0.875
-- INFO: Epoch: 0, Train loss: 0.481, Epoch time = 1123.244s
...
-- INFO: Epoch: 9, Batch[4180/4186], Train loss :0.102, Train acc: 0.984
-- INFO: Epoch: 9, Train loss: 0.100, Epoch time = 1130.071s
-- INFO: Accurcay on val 0.884
-- INFO: Accurcay on val 0.888
```

### 2.2 英文文本蕴含任务

```python
python TaskForPairSentenceClassification.py
```

运行结果:

```python
-- INFO: Epoch: 0, Batch[0/17181], Train loss :1.082, Train acc: 0.438
-- INFO: Epoch: 0, Batch[10/17181], Train loss :1.104, Train acc: 0.438
-- INFO: Epoch: 0, Batch[20/17181], Train loss :1.129, Train acc: 0.250     
-- INFO: Epoch: 0, Batch[30/17181], Train loss :1.063, Train acc: 0.375
...
-- INFO: Epoch: 0, Batch[17180/17181], Train loss :0.367, Train acc: 0.909
-- INFO: Epoch: 0, Train loss: 0.589, Epoch time = 2610.604s
...
-- INFO: Epoch: 9, Batch[0/17181], Train loss :0.064, Train acc: 1.000
-- INFO: Epoch: 9, Train loss: 0.142, Epoch time = 2542.781s
-- INFO: Accurcay on val 0.797
-- INFO: Accurcay on val 0.810
```

### 2.3 SWAG多项选择任务
```python
python TaskForMultipleChoice.py
```

运行结果：
```python
[2021-11-11 21:32:50] - INFO: Epoch: 0, Batch[0/4597], Train loss :1.433, Train acc: 0.250
[2021-11-11 21:32:58] - INFO: Epoch: 0, Batch[10/4597], Train loss :1.277, Train acc: 0.438
[2021-11-11 21:33:01] - INFO: Epoch: 0, Batch[20/4597], Train loss :1.249, Train acc: 0.438
        ......
[2021-11-11 21:58:34] - INFO: Epoch: 0, Batch[4590/4597], Train loss :0.489, Train acc: 0.875
[2021-11-11 21:58:36] - INFO: Epoch: 0, Batch loss :0.786, Epoch time = 1546.173s
[2021-11-11 21:28:55] - INFO: Epoch: 0, Batch[0/4597], Train loss :1.433, Train acc: 0.250
[2021-11-11 21:30:52] - INFO: He is throwing darts at a wall. A woman, squats alongside flies side to side with his gun.  ## False
[2021-11-11 21:30:52] - INFO: He is throwing darts at a wall. A woman, throws a dart at a dartboard.   ## False
[2021-11-11 21:30:52] - INFO: He is throwing darts at a wall. A woman, collapses and falls to the floor.   ## False
[2021-11-11 21:30:52] - INFO: He is throwing darts at a wall. A woman, is standing next to him.    ## True
[2021-11-11 21:30:52] - INFO: Accuracy on val 0.794
```

### 2.4 SQuAD问题回答任务

```python
python TaskForSQuADQuestionAnswering.py
```
运行结果：
```python
[2022-01-02 15:13:50] - INFO: Epoch:0, Batch[810/7387] Train loss: 0.998, Train acc: 0.708
[2022-01-02 15:13:55] - INFO: Epoch:0, Batch[820/7387] Train loss: 1.130, Train acc: 0.708
[2022-01-02 15:13:59] - INFO: Epoch:0, Batch[830/7387] Train loss: 1.960, Train acc: 0.375
[2022-01-02 15:14:04] - INFO: Epoch:0, Batch[840/7387] Train loss: 1.933, Train acc: 0.542
......
[2022-01-02 15:15:27] - INFO:  ### Quesiotn: [CLS] when was the first university in switzerland founded..
[2022-01-02 15:15:27] - INFO:    ## Predicted answer: 1460
[2022-01-02 15:15:27] - INFO:    ## True answer: 1460
[2022-01-02 15:15:27] - INFO:    ## True answer idx: (tensor(46, tensor(47))
[2022-01-02 15:15:27] - INFO:  ### Quesiotn: [CLS] how many wards in plymouth elect two councillors?
[2022-01-02 15:15:27] - INFO:    ## Predicted answer: 17 of which elect three .....
[2022-01-02 15:15:27] - INFO:    ## True answer: three
[2022-01-02 15:15:27] - INFO:    ## True answer idx: (tensor(25, tensor(25))
```
运行结束后，`data/SQuAD`目录中会生成一个名为`best_result.json`的预测文件，此时只需要切换到该目录下，并运行以下代码即可得到在`dev-v1.1.json`的测试结果：
```python
python evaluate-v1.1.py dev-v1.1.json best_result.json

"exact_match" : 80.530, "f1": 87.945
```
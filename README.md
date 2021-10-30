# BertWithPretrained
本项目是一个基于PyTorch实现的BERT模型及相关下游任务

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## 工程结构
- `bert_base_chinese`目录中是BERT base中文预训练模型以及配置文件

    模型下载地址：https://huggingface.co/bert-base-chinese/tree/main
- `bert_base_uncased_english`目录中是BERT base英文预训练模型以及配置文件

    模型下载地址：https://huggingface.co/bert-base-uncased/tree/main
    
    注意：`config.json`中需要添加`"pooler_type": "first_token_transform"`这个参数
- `data`目录中是各个下游任务所使用到的数据集
    - `SingleSentenceClassification`是今日头条的15分类中文数据集；
    - `PairSentenceClassification`是MNLI（The Multi-Genre Natural Language Inference Corpus, 多类型自然语言推理数据库）数据集；
- `model`目录中是各个模块的实现
    - `BasicBert`中是基础的BERT模型实现模块
        - `MyTransformer.py`是自注意力机制实现部分；
        - `BertEmbedding.py`是Input Embedding实现部分；
        - `BertConfig.py`用于导入开源的`config.json`配置文件；
        - `Bert.py`是BERT模型的实现部分；
    - `DownstreamTasks`目录是下游任务各个模块的实现
        - `BertForSentenceClassification`是单标签句子分类的实现部分；
- `Task`目录中是各个具体下游任务的训练和推理实现
    - `TaskForSingleSentenceClassification`是单标签单文本分类任务的训练和推理实现，可用于普通的文本分类任务；
    - `TaskForPairSentence`是文本对分类任务的训练和推理实现，可用于蕴含任务（例如MNLI数据集）；
- `test`目录中是各个模块的测试案例
- `utils`是各个工具类的实现
    - `data_helpers.py`是各个下游任务的数据预处理及数据集构建模块；
    - `log_helper.py`是日志打印模块；
    
## 使用方式
1. 下载完成各个数据集，并放入相应的目录中；<br>
2. 进入`Tasks`目录，运行相关模型；<br>
2.1 单文本分类任务
   
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
    2.2 文本蕴含任务
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
    
 ## 模型详细解析
- [x] [1. BERT原理与NSL和MLM](https://www.ylkz.life/deeplearning/p10631450/) <br>
- [ ] [2. BERT模型的分步实现](https://www.ylkz.life)
- [ ] [3. 基于BERT预训练模型的中文文本分类任务](https://www.ylkz.life)
- [ ] [4. 基于BERT预训练模型的英文文本蕴含任务](https://www.ylkz.life)
- [ ] [5. 基于BERT预训练模型的英文多选项任务](https://www.ylkz.life)
- [ ] [6. 基于BERT预训练模型的中文问答任务](https://www.ylkz.life)
- [ ] [7. 基于NSL和MLM任务从头训练BERT任务](https://www.ylkz.life)


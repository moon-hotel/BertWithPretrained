## 数据集下载
地址一：https://rajpurkar.github.io/SQuAD-explorer/

地址二：公众号回复"数据集"获取百度云下载地址！

数据集下载完成后，只需要将 `train-v1.1.json`,`dev-v1.1.json`这两个文件放到当前目录下即可！


## 数据格式

本目录中的`example_train.json`和`example_dev.json`为训练集和验证集的示例数据，具体介绍可以参加文章[基于BERT预训练模型的英文问答任务](../../README-zh-CN.md)。
同时，`evaluate-v1.1.py`文件是专门用来计算模型预测结果的绝对匹配率和F1值的，其使用方式如下：
```python
python evaluate-v1.1.py example_dev.json example_predictions.json
```
其中`example_predictions.json`表示模型对示例验证集预测输出的结果，形式为：
```json
{
  "56be4db0acb8001400a502ec": "[Denver Broncos]",
  "56be4db0acb8001400a502ed": "Carolina Panthers",
  "56be4db0acb8001400a502ee": "Levi's Stadium",
  "56be4db0acb8001400a502ef": "denver Broncos",
  "56be4db0acb8001400a502f0": "(gold)  "
}
```
key表示为题ID，value表示问题对应的答案。

最终预测结束后的结果通过以下方式便可以得到最终的评测指标：
```python
python evaluate-v1.1.py dev-v1.1.json best_result.json
```



## 数据集构造
见文件[data_helpers.py](../../utils/data_helpers.py)中的类 `LoadSQuADQuestionAnsweringDataset`.


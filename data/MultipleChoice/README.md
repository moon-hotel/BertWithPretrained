## 数据集下载
地址一：https://github.com/rowanz/swagaf/tree/master/data <br>
地址二：公众号回复"数据集"获取百度云下载地址！

数据集下载完成后，只需要将 `train.csv`,`val.csv`,`test.csv`这三个文件放到当前目录下即可！


## 数据格式

```python
,video-id,fold-ind,startphrase,sent1,sent2,gold-source,ending0,ending1,ending2,ending3,label
0,lsmdc1052_Harry_Potter_and_the_order_of_phoenix-94857,18313,Students lower their eyes nervously. She,Students lower their eyes nervously.,She,gold,"pats her shoulder, then saunters toward someone.",turns with two students.,walks slowly towards someone.,wheels around as her dog thunders out.,2
1,anetv_dm5WXFiQZUQ,18419,He rides the motorcycle down the hall and into the elevator. He,He rides the motorcycle down the hall and into the elevator.,He,gold,looks at a mirror in the mirror as he watches someone walk through a door.,"stops, listening to a cup of coffee with the seated woman, who's standing.",exits the building and rides the motorcycle into a casino where he performs several tricks as people watch.,pulls the bag out of his pocket and hands it to someone's grandma.,2
```

## 数据集构造
见文件[data_helpers.py](../../utils/data_helpers.py)中的类 `LoadMultipleChoiceDataset`.


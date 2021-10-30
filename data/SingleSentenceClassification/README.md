## 数据集下载
地址一：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset <br>
地址二：公众号回复"数据集"获取百度云下载地址！

## 数据格式

```python
6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
```
每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code（见下文），分类名称（见下文），新闻字符串（仅含标题），新闻关键词

分类code与名称：

```python
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
```
## 数据规模
共382688条，分布于15个分类中。

## 数据预处理
原始数据下载完成后，运行当前文件夹中的 `format.py` 脚本文件即可将原始数据安装7:2:1的比例划分成规整的训练集`train.txt`、验证集`val.txt`和测试集`test.txt`。
处理完成后的数据格式如下：
```python
轻松一刻：带你看全球最噩梦监狱，每天进几百人，审讯时已过几年_!_11
千万不要乱申请网贷，否则后果很严重_!_4
10年前的今年，纪念5.12汶川大地震10周年_!_11
怎么看待杨毅在一NBA直播比赛中说詹姆斯的球场统治力已经超过乔丹、伯德和科比？_!_3
戴安娜王妃的车祸有什么谜团？_!_2
```
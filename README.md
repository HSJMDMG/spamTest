# Naive Bayes 邮件分类

采用了朴素贝叶斯的**多项式模型**

## 1.文件结构：
* src：
  * small-email.py （email数据集处理代码）
  * large-email.py （CSDMC2010数据集的处理代码）
  * email.out （small-email.py输出文件）
  * CSDMC2010_SPAM.out(large-email.py输出文件）
* email(无数据，需要另行放入)
* CSDM2010_SPAM（无数据，需要另行放入）
* report.pdf （报告）

## 2.部分说明
* 对email数据集取10个文本交叉验证，运行100次
* 对CSDMC2010_SPAM取432（1/10）个文本交叉验证
* 输出文件内包含：
  1. 错误率error rate
  2. 分类出错的样本
  3. 把普通邮件分成垃圾邮件的次数
  4. 运行时间

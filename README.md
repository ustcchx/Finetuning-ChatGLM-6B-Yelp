## yelp_review_full数据集上lora微调ChatGLM-6B-base
### 1. 文件夹介绍
- bash：存放linux系统训练、测试、导出模型的.sh脚本
- data_process：设计prompt并处理数据集的脚本
- fig：训练时的测试集与验证集loss图像
- LLaMA-Factory：微调框架
- report：任务报告
- slurm-out：提交作业后的计算节点输出
- test-result：测试输出结果处理，其中有对F1-score指标的计算 

### 2. 微调后模型性能与微调前的性能比较
这里因为是五星制，所以认为是多分类任务，所以采用三种不同的F1-score作为评价指标。 

| F1-score    | Origin      | Finetuned   |
| ----------- | ----------- | ----------- |
| micro       | 0.406       | 0.695       |
| macro       | 0.183       | 0.579       |
| weighted    | 0.403       | 0.695       | 

可以发现，微调前后性能发生显著变化，所以我们认定微调结果是有效的。

### 3. 模型下载
[model url](https://huggingface.co/Daxuxu36/Chatglm-6B-base-Finetuning-review-rating)

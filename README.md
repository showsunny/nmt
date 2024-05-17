# nmt
pytorch implementation of neural machine translation with RNNs
pytorch实现基于RNN和注意力机制的机器翻译
![model](https://github.com/showsunny/nmt/blob/main/image/model_figure.png)
安装工具包
```bash
pip install -r requirements.txt
```
训练
```bash
source run.sh train
```
loss图像(本地训练时可以删掉第一行)
```bash
load_ext tensorboard
tensorboard --logdir runs/nmt
```
测试
```bash
source run.sh test
```

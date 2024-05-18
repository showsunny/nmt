# nmt
pytorch implementation of neural machine translation with RNNs

pytorch实现基于RNN和注意力机制的机器翻译
![model](https://github.com/showsunny/nmt/blob/main/image/model_figure.png)
## 运行步骤
克隆本仓库
```bash
git clone https://github.com/showsunny/nmt.git
```
安装工具包
```bash
pip install -r requirements.txt
```
训练
```bash
source run.sh train
```
loss图像(本地训练时可以删掉第一行,如果无法显示图像，删掉第一行，下载runs路径下生成的0文件并在本地的conda环境运行)
```bash
load_ext tensorboard
tensorboard --logdir runs/nmt
```
![loss](https://github.com/showsunny/nmt/blob/main/image/lossfig.png)
在测试集上测试
```bash
source run.sh test
```
翻译一条语句
```python
from nmt_model import NMT

import torch
import jieba

def process_jieba(text):
    words = list(jieba.cut(text))  # 转换为列表
    return words

def detokenize(tokens):
    """ Detokenize a list of tokens into a string.
    @param tokens (list[str]): List of tokens
    @returns sentence (str): Detokenized sentence
    """
    return ''.join(tokens).replace('▁', ' ').strip()

def translate_sentence(model, src_sentence):
    """ Translate a single source sentence to target language.
    @param model (NMT): Trained NMT model
    @param src_sentence (str): Source sentence
    @returns translation (str): Translated sentence
    """
    # Tokenize the source sentence
    src_tokens = process_jieba(src_sentence)
    # Perform translation
    with torch.no_grad():
        translation_hypotheses = model.beam_search(src_tokens, beam_size=5, max_decoding_time_step=70)  # Adjust beam size and max decoding time step accordingly
        # Assuming the best hypothesis is the first one
        best_translation = translation_hypotheses[0][0]
    return best_translation

def main():
    # Load the trained model
    model = NMT.load("model.bin")
    # Set the model to evaluation mode
    model.eval()

    # Example source sentence to translate
    src_sentence = "几乎已经没有地方容纳这些人, 资源已经用尽。"

    # Translate the sentence
    translation = translate_sentence(model, src_sentence)

    print("Source Sentence:", src_sentence)
    print("Translation:", detokenize(translation))

if __name__ == '__main__':
    main()
```

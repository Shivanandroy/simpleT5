<img align="center" src="data/simplet5.png" alt="simpleT5">

<p align="center">
<h3 style="text-align:center; font-weight: bold">Quickly train T5 models in just 3 lines of code with ONNX inference support
</h3>
</p>

<p align="center">
<a href="https://badge.fury.io/py/simplet5">
        <img alt="GitHub" src="https://badge.fury.io/py/simplet5.svg">
    </a>
<a href="https://badge.fury.io/py/simplet5">
        <img alt="Stars" src="https://img.shields.io/github/stars/Shivanandroy/simpleT5?color=blue">
    </a>
<a href="https://pepy.tech/project/simplet5">
        <img alt="Stats" src="https://static.pepy.tech/personalized-badge/simplet5?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads/month">
    </a>
<a href="https://opensource.org/licenses/MIT">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
</p>


**simpleT5** is built on top of PyTorch-lightning⚡️ and Transformers🤗 that lets you quickly train your T5 models.
> T5 models can be used for several NLP tasks such as summarization, QA, QG, translation, text generation, and more. 


Here's a link to [Medium article](https://snrspeaks.medium.com/simplet5-train-t5-models-in-just-3-lines-of-code-by-shivanand-roy-2021-354df5ae46ba) along with an [example colab notebook](https://colab.research.google.com/drive/1JZ8v9L0w0Ai3WbibTeuvYlytn0uHMP6O?usp=sharing) 

## Install
```python
pip install --upgrade simplet5
```


## Usage 
**simpleT5** for summarization task [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JZ8v9L0w0Ai3WbibTeuvYlytn0uHMP6O?usp=sharing)
```python
# import
from simplet5 import SimpleT5

# instantiate
model = SimpleT5()

# load
model.from_pretrained("t5","t5-base")

# train
model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
            eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len = 512, 
            target_max_token_len = 128,
            batch_size = 8,
            max_epochs = 5,
            use_gpu = True,
            outputdir = "outputs",
            early_stopping_patience_epochs = 0
            )

# load trained T5 model
model.load_model("t5","path/to/trained/model/directory", use_gpu=False)

# predict
model.predict("input text for prediction")

# need faster inference on CPU, get ONNX support
model.convert_and_load_onnx_model("path/to/T5 model/directory")
model.onnx_predict("input text for prediction")

```
## Articles
- [Geek Culture: simpleT5 — Train T5 Models in Just 3 Lines of Code](https://medium.com/geekculture/simplet5-train-t5-models-in-just-3-lines-of-code-by-shivanand-roy-2021-354df5ae46ba)
- [Abstractive Summarization with SimpleT5⚡️](https://snrspeaks.medium.com/abstractive-summarization-with-simplet5-%EF%B8%8F-344a78f73265)
- [Training T5 model in just 3 lines of Code with ONNX Inference](https://medium.com/mlearning-ai/training-t5-model-in-just-3-lines-of-code-with-onnx-inference-ff5b6678c757)
- [Kaggle: simpleT5⚡️ -  Generating one line summary of papers](https://www.kaggle.com/mathurinache/simplet5-generating-one-line-summary-of-papers)
- [Youtube: Abstractive Summarization Demo with SimpleT5](https://www.youtube.com/watch?v=jgKj-7v2UYU)

## Acknowledgements
- [Transformers by HuggingFace 🤗](https://huggingface.co/transformers/)
- [Pytorch Lightning ⚡️](https://www.pytorchlightning.ai/)
- [Fastt5](https://github.com/Ki6an/fastT5)

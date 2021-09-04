import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="simplet5",
    version="0.1.3",
    license="apache-2.0",
    author="Shivanand Roy",
    author_email="shivanandroy.official@gmail.com",
    description="simpleT5 is built on top of PyTorch-lightning ⚡️ and Transformers 🤗 that lets you quickly train your T5 models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shivanandroy/simpleT5",
    project_urls={
        "Repo": "https://github.com/Shivanandroy/simpleT5",
        "Bug Tracker": "https://github.com/Shivanandroy/simpleT5/issues",
    },
    keywords=[
        "T5",
        "simpleT5",
        "transformers",
        "NLP",
        "finetune",
        "fine-tuning",
        "pytorch",
        "summarization",
        "translation",
        "training",
        "classification",
        "Q&A",
        "inference",
        "fast inference",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=[
        "sentencepiece",
        "torch>=1.7.0,!=1.8.0",  # excludes torch v1.8.0
        "transformers==4.10.0",
        "pytorch-lightning==1.4.5",
        "tqdm"
        # "fastt5==0.0.7",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)

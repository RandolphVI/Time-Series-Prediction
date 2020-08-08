# Time-Series-Prediction

This repository contains my implementations for time series prediction task.

<!-- The main objective of the project is to predict the difficulty of each given question based on its context materials which include several components (such like document, question and option in English READING problems). -->

## Requirements

- Python 3.6
- PyTorch 1.4.0
- XGBoost 1.0.2
- Sklearn
- Numpy

## Introduction


<!-- In the widely used standard test, such as **TOEFL** or **SAT**, examinees are often allowed to retake tests and choose higher scores for college admission. This rule brings an important requirement that we should select test papers with consistent difficulties to guarantee the fairness. Therefore, measurements on tests have attracted much attention. -->

<!-- Among the measurements, one of the most crucial demands is predicting the difficulty of each specific test question, i.e., the percentage of examinees who answer the question wrong. Unfortunately, the ques- -->
<!-- tion difficulty is not directly observable before the test is conducted, and traditional methods often resort to expertise, such as manual labeling or artificial tests organization. Obviously, these human-based solutions are limited in that they are subjective and labor intensive, and the results could also be biased or misleading (we will illustrate this discovery experimentally). -->

<!-- Therefore, it is an urgent issue to automatically predict question difficulty without manual intervention. Fortunately, with abundant tests recorded by automatic test paper marking systems, test logs of examinees and text materials of questions, as the auxiliary information, become more and more available, which benefits a data-driven solution to this Question Difficulty Prediction (QDP) task, especially for the typical English READING problems. For example, a English READING problem contains a document material and  the several corresponding questions, and each question contains  the corresponding options. -->

## Project

The project structure is below:

```text
.
├── PyTorch
│   ├── LSTNet
│   │   ├── test.py
│   │   ├── layers.py
│   │   └── train.py
│   └── utils
│       ├── param_parser.py
│       └── data_helpers.py
├── data
│   ├── Train / Validation /Test_sample.json
│   ├── Train / Validation / Test_BOW_sample.json
│   └── Train / Validation / Test_pairwise_sample.json
├── LICENSE
├── README.md
└── requirements.txt
```

## Data

See data format in `/data` folder which including the data sample files. For example, `train_sample.json` is like:

```json
{"id": "6", "content": ["year", "ruined", "summer", "vacation-a", "two-week", "vacation", "wife", "family", "cabin", "lake", "northern", "ontario", "located", "boundary", "canada-by", "bringing", "modern", "convenience", "wa", "convenient", "good", "ipad", "admiring", "beauty", "nature", "checked", "e-mail", "paddling", "canoe", "twitter", "feed", "devouring", "great", "amusing", "stuck", "workday", "diet", "newspaper", "morning", "wa", "problem", "wa", "behaving", "office", "sticking", "unending", "news", "cycle", "body", "wa", "vacation", "head", "wasnt", "year", "made", "mind", "social", "medium", "experiment", "reverse", "withdrawal", "internet", "manage", "unplug", "knew", "wouldnt", "easy", "im", "good", "self-denial", "wa", "determined", "started", "physical", "restraint", "handing", "ipad", "wife", "helpfully", "announced", "wa", "read", "book", "club", "inclined", "relinquish", "tablet", "moment", "stroke", "luck", "cell", "phone", "signal", "canadian", "cabin", "wa", "spottier", "past", "making", "attempt", "cheating", "experience", "frustration", "wa", "trapped", "forced", "comply", "good", "intention", "largely", "cut", "e-mail", "twitter", "favorite", "newspaper", "website", "connect", "world", "radio-and", "radio", "listen", "choice", "planned", "read", "book", "experienced", "criminal", "plot", "street", "los", "angeles", "cutthroat", "battle", "cancer", "lab", "psyche", "london", "social", "butterfly", "magazine", "read", "im", "claiming", "cut", "internet", "completely", "day", "biked", "nearest", "town", "reward", "sat", "park", "bench", "front", "public", "library", "wi-fi", "back", "cabin", "suffered", "slow", "dial-up", "connection", "day", "check", "e-mail", "tale", "self-denial", "ha", "happy", "ending-for", "determination", "deep", "breathing", "strong", "support", "wife", "succeeded", "vacation", "struggle", "internet", "realizing", "finally", "wa", "ipad", "wa", "problem", "knew", "passed", "starbucks", "wife", "asked", "wanted", "stop", "wi-fi", "dont", "sound", "pleased", "return", "post-vacation", "situation", "test", "begin", "stay", "wagon", "im", "back", "work", "time", "compulsion", "whats", "overwhelming", "crucial", "livelihood", "intention", "giving", "membership", "cult", "immediacy", "hope", "resist", "temptation", "reflexively", "check", "e-mail", "minute", "lead", "long", "im", "checking", "twitter", "feed", "website", "vacation", "supposed", "reset", "brain", "productive", "hoping", "worked"], "question": ["doe", "underlined", "word", "restraint"], "pos_text": ["calm", "controlled", "behavior"], "neg_text": ["relaxing", "move", "strong", "determination", "unshakable", "faith"], "diff": 0.550373134328}
```

- **"id"**: just the id.
- **"content"**: the word segment of the content.
- **"question"**: The word segment of the question.
- **"pos_text"**: The word segment of the correct option.
- **"neg_text"**: The word segment of the wrong options.
- **"diff"**: The difficulty of the question.

### Data Format

This repository can be used in other similiar datasets in two ways:

1. Modify your datasets into the same format of [the sample](https://github.com/RandolphVI/Time-Series-Prediction/tree/master/data).
2. Modify the data preprocessing code in `data_helpers.py`.

Anyway, it should depend on what your data and task are.

## Usage

See [Usage](https://github.com/RandolphVI/Time-Series-Prediction/blob/master/Usage.md)

## Network Structure

<!-- Specifically, given the abundant historical test logs and text materials of question (including document, questions and options), we first design a LSTM-based architecture to extract sentence representations for the text materials. Then, we utilize an attention strategy to qualify the difficulty contribution of 1) each word in document to questions, and 2) each word in option to questions. -->

<!-- Considering the incomparability of question difficulties in different tests, we propose a test-dependent pairwise strategy for training TARNN and generating the difficulty prediction value. -->


The framework of LSTNet:

<!-- 1. The **Input Layer** comprises document representation (TD), question representation (TQ) and option representation (TO). -->
<!-- 2. The **Bi-LSTM Layer** learns the deep comparable semantic representations for text materials. -->
<!-- 3. The **Attention Layer** extracts words of the document (or the option) with high scores as dominant information for a specific question, which is helpful for visualizing the model and improving the performance. -->
<!-- 4. Finally the **Prediction Layer** shows predicted difficulty scores of the given READING problem. -->

## Reference

**If you want to follow the paper or utilize the code, please note the following info in your work:**

- **Model LSTNet**

```bibtex
@inproceedings{lai2018modeling,
  title={Modeling long-and short-term temporal patterns with deep neural networks},
  author={Lai, Guokun and Chang, Wei-Cheng and Yang, Yiming and Liu, Hanxiao},
  booktitle={The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval},
  pages={95--104},
  year={2018}
}
```

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
# ICLR 2026 Acceptance Prediction: Benchmarking Decision Process with A Multi-Agent System

<font size=3><div align='center' > [[🍎 Project Page](https://video-mme.github.io/)] [[🎯 ICLR 2026 Prediction](https://huggingface.co/datasets/simon123905/ICLR/blob/main/ICLR_2026_Prediction.zip)] [[🤗 Dataset](https://huggingface.co/datasets/simon123905/ICLR/tree/main)] </div></font>

## 🔥 News
* **`2026.01.19`** 🌟 We’ve released ICLR 2026 predictions powered by the PaperDecision framework—check them out now!

## 👀 PaperDecision Overview
Academic peer review is central to research publishing, yet it remains difficult to model due to its subjectivity, dynamics, and multi-stage decision process. We introduce <strong>PaperDecision</strong>, an end-to-end framework for modeling and evaluating the peer review process with large language models (LLMs).

Our work distinguishes itself through the following key contributions:

* *An agent-based review system.* We develop <strong>PaperDecision-Agent</strong>, a multi-agent framework that simulates authors, reviewers, and area chairs, enabling holistic and interpretable modeling of review dynamics.

* *A dynamic benchmark.* We construct <strong>PaperDecision-Bench</strong>, a large-scale multimodal benchmark that links papers, reviews, rebuttals, and final decisions, and is continuously updated with newly released conference rounds to support forward-looking evaluation and avoid data leakage. 

* *Empirical insights.* We achieve up to <strong>~82% accuracy</strong> in accept–reject prediction with frontier multimodal LLMs and identify key factors influencing acceptance outcomes, such as reviewer expertise and score changes.

<p align="center">
    <img src="./asset/teaser.png" width="100%" height="100%">
</p>

## 🤖 PaperDecision-Agent
<strong>PaperDecision-Agent</strong> models key roles in the real-world peer review process through specialized agents. It simulates structured interactions among authors, reviewers, and area chairs to capture the full decision-making workflow.

<p align="center">
    <img src="./asset/workflow.png" width="100%" height="100%">
</p>

## 📊 

## :black_nib: Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
```

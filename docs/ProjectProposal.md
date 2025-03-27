# Project Proposal

## Project Introduction

- Title: "Audio" Book Annotator and Summarizer
- Team member: Nam and Google Colab

## Project Description

This project aims to provide a summarization of both books and audiobooks while also giving credits back to the writer/voice behind the books itself and avoid hallucinations by actively tracing back to the original materials.
There are 3 main steps:

- Book summarization using a Finetuned version of a Pretrained model to provide the best summarization with supporting verbatim context in minimal time (Models: Gemma 3 4B, Qwen 2.5 3B)
- Audio Book Transcribing using WHISPER or an extension of WHISPER
- Collect information to create the wanted result either using direct querying in a vector DB for future scaling or direct cutting from audio file
- Optional: KV Caching

## Algorithms

- GRPO
- Supervised Finetuning

### 1. GRPO vs SFT:

#### SFT

Supervised Fine-Tuning (SFT) involves training a pre-trained language model on a dataset of input-output pairs. In the context of summarization, the inputs are the book chapters or text segments, and the outputs are the corresponding summaries. The model is trained to minimize the difference between its generated summaries and the reference summaries using a loss function like cross-entropy. SFT is effective for adapting a pre-trained model to specific tasks when a high-quality labeled dataset is available.

#### GRPO

GRPO, or Gradient Ranked Preference Optimization, is a reinforcement learning technique that optimizes a language model based on pairwise preferences rather than direct supervision. It utilizes a reward model to score generated summaries and then updates the language model to generate summaries that are ranked higher by the reward model. This method is particularly useful when explicit reference summaries are scarce or when the desired output is subjective and difficult to define precisely. GRPO allows for fine-tuning the model based on relative quality comparisons, making it robust to variations in summarization styles.

#### Why these two methods?

We chose SFT as a foundational method due to its proven effectiveness in summarization tasks. The Booksum-randomized dataset provides high-quality input-output pairs, making SFT a suitable choice for initial model training. GRPO is selected to potentially enhance the model's performance by incorporating a reward-based learning approach. This method can help refine the model's summarization style and improve its ability to generate summaries that are both accurate and preferred by human evaluators. By comparing the results of SFT and GRPO, we can assess the benefits of each approach and potentially combine them for optimal performance or pick out one.

## Dataset

[Booksum-randomized](https://huggingface.co/datasets/nschantz21/booksum-randomized/viewer/default/train?row=0&sql=SELECT+AVG%28chapter_length%29+AS+mean_chapter_length%0AFROM+train%3B&views%5B%5D=train&views%5B%5D=validation)

- This is a version of the CMU Book Summary dataset but also have the chapter text.

### 1. Reason for choice of dataset

The Booksum-randomized dataset is chosen because it provides both the chapter text and summaries, which are essential for training and evaluating our summarization model. The inclusion of chapter text allows for granular summarization and enables the creation of detailed annotations. The randomization ensures a diverse training set, mitigating potential biases.

## Course-Related Topics

- KL Divergence
- Loss Function
- Reinforcement Learning
- Evaluation and Performance Measurement
- Supervised Finetuning (Training)

## Timeline

March 15th: Project initialization and start to automate classification and chunking
March 20th: Start training summarizer model
March 27-30th: Finalize and compare with Base Model and some other OSS models
April 1st: Transcribe Audio Book to generate timestamp with text of the original audio.
April 4th: Setup Vector Database to store the subtitles with timestamp and test different querying strategies
[May be also setup a vector database to store the chapter summaries and whole book summaries]
April 7th: Clean up, set up pipeline, evaluation and plan for extending capabilites

## Evaluation plan

### 1. ROUGE Score

#### a. Overview of algorithm

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used for evaluating automatic summarization and machine translation. It works by comparing an automatically produced summary or translation against a set of reference summaries (human-produced summaries). The metrics count the number of overlapping units, such as n-grams, word sequences, or word pairs, between the generated summary and the reference summaries. Common ROUGE metrics include:

- **ROUGE-N**: Measures the overlap of n-grams between the generated and reference summaries.
- **ROUGE-L**: Measures the longest common subsequence (LCS) between the generated and reference summaries.
- **ROUGE-W**: Weighted LCS-based similarity.
- **ROUGE-S**: Skip-bigram co-occurrence statistics.

#### b. Reason for choice

ROUGE is a widely accepted and established metric for evaluating summarization tasks. Its recall-oriented nature makes it particularly suitable for assessing how well the generated summaries capture the key information from the reference summaries. Using ROUGE allows for a quantitative comparison of the model's performance against human-generated summaries.

### 2. Modified BLEU

#### a. Overview of algorithm

BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. The original BLEU metric focuses on precision, measuring how much the words (or n-grams) in the generated text appear in the reference text. However, for summarization, recall is often more important. Therefore, a "modified" BLEU can be used to incorporate recall and handle shorter summaries more effectively.

Modifications may include:

- **Clipping**: Limiting the count of n-grams to the maximum count in any single reference summary.
- **Brevity Penalty**: Penalizing summaries that are too short.
- **Weighted N-grams**: Adjusting the weights of different n-grams to better reflect their importance.
- **Recall based modifications**: changing the focus from precision to recall.

#### b. Reason for choice

While BLEU is traditionally used for translation, modified versions can provide valuable insights into the quality of generated summaries. By adjusting the metric to emphasize recall and account for brevity, it can complement ROUGE in evaluating the model's ability to capture relevant information. Additionally, using both ROUGE and a modified BLEU provides a more comprehensive evaluation, considering both precision and recall aspects of summarization.

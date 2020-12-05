---
layout: post
title:  "Attention is all you need - The Transformer"
tags: [ Tips, Natural Language Processing, Neural Networks  ]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
---

This page is in development. Please come back later.


### Background: 

Historically RNN's like LSTM's and GRU's are widely used architectures for most of natural language understanding tasks like Machine traslation, language modelling. They performed decently well but one major drawback is RNN's process data sequentially. This inhibits parallelization, so more words we have in input sequence, more time it will take to process that sentence. In addition to sequential nature of RNN's, they suffer from vanishing gradient problems for long sequences. **Transformer** was introduced in paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) and it was shown that this new architecture is able to tackle the problems faced by RNN's

I will try to explain each and every detail of Transformers in this blog post. Contents are as follows

1) What is Transformer?
2) Overview of Architecture
3) Encoder
  * 


### What is Transformer?

The Transformer a architecture whose main goal is to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It totally can take advantage of parallelization and relies on [attention mechanism](https://arxiv.org/pdf/1508.04025v5.pdf) to draw global dependencies between input and output

In [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, transformer was used mainly for machine translation. So lets keep our discussion in that domain when going through different components. Lets say for example our goal is to translate english sentence in to french sentence


### Overview of Architecture:

Transformer contains two main components.

1) Encoder
2) Decoder

Input to Encoder would be English sentence and decoder outputs corresponding french sentence. Below is simple diagram showing how transformer is structured


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_4.png" />
</p>


Looking at above picture, one might say that this looks exactly like sequence to sequence machine translation using RNN's where encoder takes input and decoder produces output. But all the magic happens inside big boxes above encoder/decoder where instead of sequential processing we have parallel processing and relies on 3 types of attention. So lets have a look at each component in detail.


### Encoder:

Encoder block is a stack consisting of 6 identical layers. Each layer has 2 sublayers. First is a multi head self attention mechanism and second is a simple position wise fully connected feed forward network. Let discuss with a simple picture


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_5.png" />
</p>


As seen above, encoder's input first flow through a self attention layer that helps the encoder look at other words in the input sentence as it encodes specific word. The outputs of self-attention layer are fed to feed-forward neural network. It is applied independently to each position.


Main property of transformer is that each word in input flows throgh its own path in the enoder and in self attention these input words interact and have dependencies between them. Feed forward layer do not have these interactions and since each word flows in its own path, entire sequence can be executed in parallel as shown below.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_6.png" />
</p>


Lets understand all the operations that happen in encoder with the help of 3 examples

1) They go to gym everyday
2) He is studying in third grade
3) Red cat sitting on the table


As with any NLP task, we first tokenize the sentence and convert in to numbers. After that we convert each number(corresponding to a word) in a sentence in to word embeddings. First will see how one sentence flows through encoder using vectors and slowly will transition in to using matrices for batch of sentences


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_7.png" />
</p>


Above I used embedding size of 4 for illustration. But in paper they used embedding size of 512. So input will be a list of vectors each of size 512. First input passes through Self Attention layer and then through Feed forward neural network layer







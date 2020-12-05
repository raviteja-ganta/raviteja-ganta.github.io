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


#### Encoder:

Encoder block is a stack consisting of 6 identical layers. Each layer has 2 sublayers. First is a multi head self attention mechanism and second is a simple position wise fully connected feed forward network. Let discuss with a simple picture


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_5.png" />
</p>



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


#### Positional encoding



#### 



Now we know how one sentence pass through encoder, lets see how batch of sentences flows. This is very important becauase in real world we never deal with one sentence. Below is the simple picture that illustrates this. Again I used toy embedding of size 4 for illustration.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_8.png" />
</p>


Above input representation flows through self attention layer in encoder. This is the heart of transformers. So lets dig deeper


#### Self-Attention


Consider the sentence *I arrived at the bank after crossing the...*. In this sentence depending on the word which ends the sentence, meaning and appropriate representation of word **bank** changes. For example if ending word is river then this word **bank** refers to bank of the river and if ending word is let's say for example **road** then it refers to finanical instititution. So when we are re-representing a particular word, model needs to look at surrounding words in input sentence for clues that can help lead to a better encoding for this word.


> Self Attention is the mechanism Transformer uses to bring in information of other relevant words in to the one we are currently processing


So in above example, word **river** would recive high score i.e, when processing word bank to compute new representation, our model would give high attention to word river. We can think of the final score a particular word would get is a weighted combination of representations from all surrounding words.


For each word in input sentence, we want to get a score of all remaining words in that sentence. So some how we need to compare each word with all others. This is the main computation involved in self-attention. We want to get something like below



<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_9.png" />
</p>



#### Self-Attention calculation:

Lets see now how can we calcualte similarity scores as in above fig.

Instead of comparing embedding vectords directly, embeddings are first transformed using two linear layers. One such transformation generates query tensor(Q), the other transformation leads to key tensor(K). Since we are doing self-attention, query and key corresponds to same input sentence.


##### Creating Query(Q), Key(K) and Value(V) matrices

Input representation(Fig.5) is multiplied by WQ matrix(linear layer) to get Q and multiplied by WK matrix to get K. Fig.5 has dimensions of (n_seqXembed_dim) = (6X4). We will be using WQ/WK matrix of size 4X4 for illustration. Q/K matrices will have shape 6X4 as shown below. For self attention Q = K


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_10.png" />
</p>


We are projecting these input vectors in to a space where dot product is a good proxy for similarity. Higher the dot product score, more similar or more attention between words.


Attention scores are calculated by dot product between Q and K matrices i.e. Q<sup>T</sup>K as shown below


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_11.png" />
</p>


If we see above attention scores are large numbers. Now we apply softmax function per row to bring all these numbers to between 0 and 1. Also good thing is that they add up to 1 for any particular word.


We have introduced Query and Key tensors so far. The last piece missing until now is the Value tensor. Valuen tensor is formed by a matrix multiply of the Input representation(Fig.5) with the weight matrix WV whose values were set by backpropagation. The row entries of V are then **related** to the corresponding input embedding. 

The purpose of the dot-product is to 'focus attention' on some of the inputs. Dot product which is softmax(Q<sup>T</sup>K) now has entries appropriately scaled to enhance some values and reduce others. These are now applied to the V entries.

The matrix multiply weights first column of V, representing a section of each of the input embeddings, with the first row of softmax(Q<sup>T</sup>K), representing the similarity of word#0 and each word of the input embedding and deposits the value in Z


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_12.png" />
</p>


With above calculation we are re-representing each word in a sentence by taking in to account all surrounding words information. This is called self-attention. Its **self** because when encoding a particular sentence we are just looking at all other words in that sentence including itself but no where we are referring to other sentences.


#### Multi-head Attention:

Sometimes its beneficial to get multiple representational subspaces for a word. For example in our sentence *I arrived at the bank after crossing the river* when we calculated our representation Z, even though it uses all other words information to re-represent word **bank**, it could be dominated by the the actual word itself. So authors of paper thought why not use multiple attention heads(which could be applied in parallel using multiple sets of Query/Key/Value weight matrices). Lets try to understand this with a simple example where we used 2 attention heads instead of 1 head which we were used to until now



<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_13.png" />
</p>


How do we calculate this? Its very simple instead of using one set of Query/Key/Value weight matrices to produce one set of Q,K,V matrices, we use multiple set of Query/Key/Value weight matrices to produce multiple set of Q,K,V matrices. All these calculations will be done in parallel. Lets go step by step using number of attention heads = 2(paper used 8 attention heads)


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_14.png" />
</p>


Now we perform **self-attention** using 2 heads instead of 1 head as we are doing until now. As we know first step is to calculate attention scores using dot product between Query(Q) and Key(K) vectors as shown below


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_15.png" />
</p>


Now we have 2 sets of attention scores corresponding to 2 attention heads, we need to multiply these with Value(V) matrices as before to get final output of self-attention


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_16.png" />
</p>






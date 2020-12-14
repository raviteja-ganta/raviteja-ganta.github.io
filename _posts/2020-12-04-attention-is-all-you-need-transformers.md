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


After applying multi-head attention each word in a sentence will have multiple representations(actually n_head representations where n_head is number of attention heads). But our feed forward layer is expecting one matrix as show in Fig 9 but we have two matrices now in Fig 13. So we need a way to condense this information to one metric. This is done by first concatenating the two metrices and passing through linear layer as shown below


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_17.png" />
</p>



**Entire self-attention process is described below**


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_18.png" />
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_19.png" />
</p>



#### Feed forward Neural network layer

Output of self attention layer will flow through FFNN as shown in Fig 4. FFNN is applied to each position seperately and identically and it consists of 2 layers as shown below.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_20.png" />
</p>


#### Residual connections and LayerNorm

Until now we had a look at 2 main components of a encoder layer. Transformer architecture had 6 layers like this and output of one layer will act as input to other layer. But there is one thing which needs to be discussed which is residual connections. We have residual connection around each of two sub layers in encoder layer then followed by layer normalization. Lets understand this with an example below. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension dmodel = 512. Below image uses toy dimensions of 4 for illustration


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_21.png" />
</p>


### Decoder

Decoder is right most part of architecture that decodes the encoder's encoding of input sentence. Decoder works exactly same as encoder except that it has one more attention layer called Encoder-decoder attention. Below is overview of decoder layers.

1) Causal self attention - In a sentence, words only can look at previous words when generating new words. Causal attention allows words to attend to other words that are related, but important thing to keep in mind is, they cannot attend to words in the future since these were not generated yet, they can attend to any word in the past though. This is done by masking future positions.

2) Encoder-decoder attention - For example lets say we are translating english to french. Input to encoder would be english sentence and decoder outputs french sentence. Encoder-decoder attention is the one which helps decoder focus its attention on appropriate places of input sentence. In this layer Queries come from French sentence/previous decoder layer but Values and keys come from output of final encoder layer. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models.

3) Feed forward layer - FFNN is applied to each position seperately and identically similar to encoder side.

4) Residual and LayerNormalization - Similar to encoder, decoder also has residual connections and layer normalization.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_22.png" />
</p>


#### Inputs and Outputs in decoder

Outputs in transformer are generated token by token and the most recently generated token will be used as input for next time step. This mimics the language modelling generation as shown below.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_23.png" />
</p>


#### Causal self attention

Causal self attention allows words to attend to other words that are already generated when generating new word. For example in Fig 19 when generating word *belle* causal self attention allows to attend to only words *C'est* and *une* but not to word *matinee* since matinee is not generated yet.

Since it is just a variant of self attention, queries and keys come from same sentence.

Causal self attention works same way as self attention of Encoder except that we need to modify calculation little to take care of above point. This is done by masking future positions. Lets understand this with one sentence but logic would hold true even for batch of sentences.

English sentence: They go to gym everyday

French translation: Ils vont au gym tous les jours


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_24.png" />
</p>

**Step by step:**

1) Similar to self-attention we calcuate Q<sup>T</sup>K using French sentence which is matrix C in above fig. Queries(Q) and key(k) both come from same sentence which is French sentence here.

2) Matrix C says how much similar each word in query is to each word in key. This is called attention weight matrix. For example pink strip in matrix C above says how much similar *vont* in query is to each word in key. So far its similar to self-attention. But thing is when we are generating a word in query, we only want to look at words that were generated until then including itself. So for example when generating word *vont*, this word only wants to look at words *Ils* and *vont*. But our similarity matrix has similarity scores even for words that come later. So we want to mask these similarity scores for words which come later.

3) This is done by adding matrix M which has all zeros on diagonal and below and large negative numbers above diagonal. This gives matrix f. Idea is that when we apply softmax in next step these large negative numbers becomes zero as show below.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_25.png" />
</p>

 
So from matrix a in fig 21 above, when generating word *vant* it has non zero similarity values only for words *Ils* and *vant*. This helps decoder to pay attention to only words that were generated in the past.


Steps that come after the masking were exactly the same as in multi-headed self attention we discussed on encoder side.


#### Encoder-decoder attention

This is very similar to self-attention in encoder except that in this case Queries(Q) come from previous decoder layers, Keys(K) and values(V) come from output of final encoder layer. This allows every position in the decoder to attend over all positions in the input sequence.


### Final linear layer and softmax

Output of decoder will flow through linear layer and through softmax to turn a vector of words in to actual word. These layers are applied on each individual position if decoder seperately and in parallel as shown below

Linear layer helps to project output from decoder layers in to higher dimension layer(this dimension depends on vocablary). For example if our french vocablary has 20000 words then this linear layer will project our decoder output of any single token in to 20000 dimension vector. Then this 20000 dimensional vector will be passed through softmax layer to convert in to probabilities. Now we select the token with highest probability as the output word at this token position.


<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Transformers/tf_26.png" />
</p>


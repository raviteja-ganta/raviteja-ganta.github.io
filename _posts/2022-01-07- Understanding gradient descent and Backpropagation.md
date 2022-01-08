---
layout: post
title:  "Understanding gradient descent and Backpropagation"
tags: [ Tips, Neural Networks]
featured_image_thumbnail: assets/images/Transformers/tf_1_thumbnail.jpg
featured_image: assets/images/Transformers/tf_2.jpg
---

*This post is under development. Please come back later

Gradient descent and backpropagation are workhorse behind training neural networks and understanding what's happeing inside these algorithms is atmost importance
for efficient learning. This post gives in depth explanation of gradient descent and Backpropagation for training neural networks.


Below are the contents:

1) Notation to represent neural network
2) Forward propagation



1) Notation to represent a neural network:

I will be using a 2 layer neural network as shown below through out this post as our running example.

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp1.png" />
</p>

*w<sub>jk</sub><sup>[l]</sup>* - Weight for connection from k<sup>th</sup> neuron in layer (l-1) to j<sup>th</sup> neuron in layer l

*b<sub>j</sub><sup>[l]</sup>* - Bias of j<sup>th</sup> neuron in layer l

*a<sub>j</sub><sup>[l]</sup>* - Activation of j<sup>th</sup> neuron in layer l

2) Forward propagation:

Below is the calculation that happens as part of forward propagation in single neuron

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp2.png" />
</p>

So the activation at any neuron in layer l can be written as 

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp4.png" />
</p>

<!-- <img src="https://render.githubusercontent.com/render/math?math=\LARGE a_j^[l]"> = <img src="https://render.githubusercontent.com/render/math?math=\LARGE f"><img src="https://render.githubusercontent.com/render/math?math=\LARGE ("><img src="https://render.githubusercontent.com/render/math?math=\displaystyle \sum_{k}"> <img src="https://render.githubusercontent.com/render/math?math=\LARGE w_jk^[l] a_k^[l-1]">   -->

Matrix representation:

For Neural network in Fig 1 above,

*w<sup>[l]</sup>* - Weight matrix for layer *l*

We use equation 1 from above to calulate activations for every layer but all calculations are done using matrix mulitiplcations as they are very fast

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Backprop/bp3.png" />
</p>


Main goal of backpropagation is to understand how varying individiual weights of network changes the errors made by the network. But how do we calculate error made by the nextwork. We use term called cost function to get estimate of error. Cost gives estimate of how bad or good our network predictions are. This is function of actual label and predicted value

$$c_x = f(y^i,yhat)$$

where <img src="https://render.githubusercontent.com/render/math?math=\LARGE y^i"> is the actual label or truth of data point *i* and yhat is the prediction from neural network.

We usually get cost for individual training example and overall cost would be average of cost over all the training examples. Written as 

$$C = \frac{1}{n} \sum_{x} c_x$$

Lets use quadratic cost for illustration in our example which is $$c_x = \frac{1}{2} (y^i - yhat)^2$$. Here yhat would be exactly same as activations/output of final layer of neural network

$$c_x = \frac{1}{2} (y^i - a^{[L]})^2$$

$$a^{[L]}$$ is a vector as shown in fig.3 and L = 2 for our example neural network of fig.1. Since $$y^i$$ is fixed for any training example, so cost $$c_x$$ is function of $$a^{[L]}$$

> Back propagation is all about understanding how changing the weights and biases in a network changes the cost function $$c_x$$


Using Back propagation we calculate $$\frac{\partial c_x }{\partial w}$$ and $$\frac{\partial c_x }{\partial b}$$ for single training example and then from this we can calculate $$\frac{\partial C }{\partial w}$$ and $$\frac{\partial C }{\partial b}$$ by overaging over training examples. If we see per node, then our goal is to find $$\frac{\partial C }{\partial w_{jk}^{[l]}}$$ and $$\frac{\partial C }{\partial b_{j}^{[l]}}$$ for any layer *l* . 

### Contribution of error from a single neuron

Lets say we change our parameters and biases coming in to a neuron *j* of layer *l*. We know forward propagation equation as below. *k* represents all the *k* nodes of previous layer *l-1*

$$z_j^{[l]} = \sum_{k} w_{jk}^{[l]} a_{k}^{[l-1]} + b_{j}^{[l]}$$

So as we change parameters $$w_{jk}^{[l]}$$ and $$b_{j}^{[l]}$$ corresponding to any neuron *j* of layer *l* then output $$z_j^{[l]}$$ generated in that neuron also changes and this in turn finally change our cost function $$c_x$$. We can write this change as $$\frac{\partial c_x }{\partial z_{j}^{[l]}}$$. So we can define contribution of change/error from this single neuron as 

$$e_j^{[l]} = \frac{\partial c_x }{\partial z_{j}^{[l]}}$$

Using Backpropagation we can compute $$e_j^{[l]}$$ first and then derive $$\frac{\partial c_x }{\partial w_{jk}^{[l]}}$$ and $$\frac{\partial c_x }{\partial b_{j}^{[l]}}$$ from this error.

### Main equations of Backpropagation

#### Equation 1 - Equation for error in output layer $$e_j^{[L]}

$$e_j^{[L]} = \frac{\partial c_x }{\partial a_{j}^{[L]}} g'_{\text{activ}}(z_{j}^{[l]})$$




$$
\begin{align}
\frac{\partial E }{\partial w_{jk}} &= \frac{1}{2} \sum_{k}(a_k - t_k)^2 \\  
&= (a_k - t_k)\frac{\partial}{\partial w_{jk}}(a_k - t_k) \tag{2}
\end{align}
$$

$$ x = y^2 $$

In N-dimensional simplex noise, the squared kernel summation radius $r^2$ is $\frac 1 2$
for all values of N. This is because the edge length of the N-simplex $s = \sqrt {\frac {N} {N + 1}}$
divides out of the N-simplex height $h = s \sqrt {\frac {N + 1} {2N}}$.
The kerel summation radius $r$ is equal to the N-simplex height $h$.

$$ r = h = \sqrt{\frac {1} {2}} = \sqrt{\frac {N} {N+1}} \sqrt{\frac {N+1} {2N}} $$



---
layout: post
title:  "Understanding Image style transfer using Convolutional Neural Networks"
tags: [ Tips, Computer Vision  ]
featured_image_thumbnail: assets/images/Neural_style_transfer/NS_fig1_thumbnail.png
featured_image: assets/images/Neural_style_transfer/NS_fig1.png
---

Main goal of this post is to explain Gatys et al (2016) work on Image style transfer using CNN’s in easier terms.

Code for generating all images in this notebook can be found at [github](https://github.com/raviteja-ganta/Neural-style-transfer-using-CNN)

### Contents:

1) What is Style transfer?

2) Content of an Image

3) Style of an Image

4) Understanding output of CNN’s

5) Cost function

6) Gram matrix

### 1) What is Style transfer?

First of all, what is style transfer between images? I will try to explain it with the example above.

We have content image which is a stretch of buildings across a river. We also have a style image which is a painting. Main idea behind style transfer is to transfer the ‘style’ of style image to the content image so that the target images looks like buildings and river painted in style of artwork(style image). We can clearly see that content is preserved but looks like buildings and water are painted.

To do this we need to extract content from content image, style from style image and combine these two to get our target image. But before that, lets understand what exactly content and style of an image are.

### 2) Content of an Image:

Content can be thought as objects and arrangements in an image. In our current case, content is literally content in the image with out taking in to account texture and color of pixels. So in our above examples content is just houses, water and grass irrespective of colors.

### 3) Style of an Image:

We can think of style as texture, colors of pixels. At same time it doesn’t care about actual arrangement and identity of different objects in that image.

Now that we have understanding of what content and style of image are, lets see how can we get them from the image.

But before that its important to understand what CNN’s are learning. It gives us clear idea when we talk about extracting style from image.

### 4) Understanding output of CNN’s:

I will be using trained Convnet used in paper Zeiler and Fergus., 2013, Visualizing and understanding convolutional networks and visualize what hidden units in different layers are computing. Input to the below network is ImageNet data spread over 1000 categories.

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig2.png" />
</p>
<p style="text-align: center;">*Fig. 2*</p>

Lets start with a hidden unit in layer 1 and find out the images that maximize that units activation. So we pass our training set through the above network and figure out what is the image that maximizes that units activation. Below are the image patches that activated randomly chosen 9 different hidden units of layer 1
 
<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig3.png" />
</p>

Fig. 3 (a) gives sense that hidden units in layer 1 are mainly looking for simple features like edges or shades of color. For example, first hidden unit(Row1/Col1) is getting activated for all 9 images whenever it see an slant edge. Same way Row2/Col1 hidden unit is getting activated when it sees orange shade in input image.<br/>
 
<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig4.png" />
</p>

Zeiler and Fergus visualized same for deeper layers of Convnet with help of deconvolutional layers. For layer 2 looks like it detecting more complex shapes and patterns. For example R2/C2 hidden unit is getting activated when it sees some rounded type object and in R1/C2 hidden unit is getting activated when it see vertical texture with lots of vertical lines. So the features second layer is detecting are getting more complicated.

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig5.png" />
</p>

Zeiler and Fergus did same experiment for layer 5 and they found that its detecting more sophisticated things. For example hidden unit(R3/C3) is getting activated when its sees a dog and hidden unit(R3/C1) is maximally activated when it see flowers. So we have gone long way from detecting simple features like edges in layer 1 to detecting very complex objects in deeper layers.

### 5) Cost function:

In order to do neural style transfer we define a cost function to see how good the generated image is. We can use gradient descent to lower this cost by updating the generated image until generated image is what we want. We have two cost functions 1) Content cost : Measures how similar content of generated image is to content of content image 2) Style cost: Measures how similar style of generated image is to style of style image

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig6.png" />
</p>

Goal for above cost function is to take a target image which usually we start as random noise or as a copy of content image and change it so that content is close to content image and style is close to style image

#### Content cost function:
As we saw from above research by Zeiler and Fergus, as we go deeper in to CNN, later layers are increasingly care about content of image rather than texture and color of pixels(Images shown above are not actual output of CNN layers so the reason they are colored). Authors used features from pretrained VGG19 network for extracting both content and style of an image. For content cost, both content and target image are passed through VGG19 pretrained network and output of Conv4_2 is taken as content representation of image.

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig7.png" />
</p>

Lets name P and F as content representations(output of Conv4_2 layer) of content and target image respectively. If these two are equal then we can say that contents of both content image and target image are matching. So content cost is how different are these representations(Cc and Tc). We just take element wise difference between hidden unit activations between Cc and Tc

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig8.png" />
</p>

Our goal is to minimize above loss by changing the target image using gradient descent updating its appearance until its content is similar to that of content image. This is similar to minimizing classification loss but here we are updating target image and not any filters or coefficients of model.

#### Style cost function:
To obtain a representation of the style of an input image, authors used a feature space designed to capture texture information. Style is calculated as correlation between activation's across different channels or in other words style representation of image relies on looking at correlations between different channels in a layer output.

But why does this represent style? For explanation lets use R1/C2 neuron and R2/C1 neuron of Fig. 3(b) as example and assume these two neurons represents two different channels of layer 2. R1/C2 neuron is getting highly activated when in input image it sees fine vertical textures with different colors and R2/C1 neuron is getting activated when it sees orange colors. So what does it mean these two channels to be highly correlated? It means for same part of image, vertical texture and orange colors occur together. So correlation tells us which of these high level texture components occur or do not occur together. So for example, we found that correlations between these two channels is high whenever style image passes through them. Since these two channels are specialized in finding vertical textures and orange colors respectively and if correlations between these two channels are high even when target image is passed then we can say that style of both images are identical with respect to these two channels.

### 6) Gram matrix

Correlations at each layer is given by gram matrix. Each position of a gram matrix for a layer gives value of correlation between two different channels in that layer. Below is the calculation of style loss for one layer

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig9.jpeg" />
</p>
->*Fig. 5: Explanation of gram matrix*<-

Authors of paper included feature correlations of multiple layers to obtain multi scale representation of input image, which captures texture information but not global arrangement.

I used Conv1_1, Conv2_1, Conv3_1, Conv4_1, Conv5_1 layers to get style loss. They are weighed for final style loss. I gave higher weight for Conv1_1 and Conv2_1 as we have seen above that earlier layers are ones that catches texture patterns. Overall style cost is as below. Again we will only change target image to minimize this below loss using gradient descent.

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig10.png" />
</p>

Again in calculation of final loss we have coefficients alpha and beta. Authors of paper used alpha/beta ratio in range of 1* 10−3 to 1* 10−4. Lower the value of this ratio, more stylistic effect we see. But for my generated image which we saw at start of this blog, I used ratio of 1* 10–7 as different ratios work well for different images.

So goal of the problem is to modify target image over number of iterations of gradient descent to minimize combined cost function.

Below is one more example of style transfer

<p align="center">
  <img src="https://raw.githubusercontent.com/raviteja-ganta/raviteja-ganta.github.io/main/assets/images/Neural_style_transfer/NS_fig11.png" />
</p>

Any inputs to make this story better is much appreciated.

### References:

* Image Style Transfer Using Convolutional Neural Networks. Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
* Visualizing and Understanding Convolutional Networks. Matthew D Zeiler, Rob Fergus
* Deep learning specialization by Andrew Ng
* Deep learning engineer Nano degree Udacity.

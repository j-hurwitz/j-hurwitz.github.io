---
layout: post
title:  neural net framework from scratch, spelled-out tutorial
date:   2023-09-05
categories: jekyll update
use_math: true
---
# Part 1. In Theory, In Practice, The Diagram

In theory, the artifical neural network is incredibly powerful. This is because a neural net can learn almost anything. Well, it's more correct to say a neural net can compute any function. This is a function in the math sense, $f: R^n \rightarrow R^m$ which maps a sequence of $n$ real inputs to a sequence of $m$ real outputs. Well, it's more correct to say that a neural net is only *guaranteed* to *approximate* any function. The good news is this approximation can have arbitrarily low error. There are some more formal math considerations to make the statement precise, but they all amount to a math theorem called the Universal Approximation Theorem. Here is a great [visual explanation](http://neuralnetworksanddeeplearning.com/chap4.html) of how a neural net can represent complex functions from very simple building blocks. It assumes that the neural network can be arbitrarily large, but still, given any function, the most basic kind of neural network can approximate that function to any arbitrary accuracy. What kind of functions could we learn?
- given some text, predict what text [should come next](https://chat.openai.com/)
- given dashcam video of a car driving, predict how the gas/brakes/steering wheel should behave to [drive like the average human](https://www.comma.ai/openpilot)
- given medical data, does patient have a disease or not

Luckily we generally don't need to worry about the precise mathematical details of this "unknown function" that we are trying to approximate, and we can just collect training data, try to train a neural net on the data, and see how well we can do.

In practice, the neural network is incredibly powerful. Before 2015, some experts in the ancient Chinese board game of Go thought that computers would never beat the world Go champion. Some thought it was possible, but that we were many years away from such a feat. Then a neural net called [AlphaGo](https://www.youtube.com/watch?v=WXuK6gekU1Y) beat the world champion 4-1, exhibiting superhuman skill and alien-like strategies to Go experts. An extension of that approach produced AlphaZero which can learn to play Go, Chess, Shogi, and Atari games, and achieve superhuman performance in all of them. The same company, DeepMind, then created AlphaFold which won the "world championship" of protein-folding prediction, blowing the competition (teams of expert scientists) out of the water in accuracy. You also may have heard of ChatGPT, the neural network.

<svg width="500" height="300">
        <!-- Draw input layer -->
        <circle cx="50" cy="50" r="10" stroke="black" stroke-width="2" fill="green"></circle>
        <circle cx="50" cy="150" r="10" stroke="black" stroke-width="2" fill="green"></circle>
        <circle cx="50" cy="250" r="10" stroke="black" stroke-width="2" fill="green"></circle>

        <!-- Draw hidden layer -->
        <circle cx="200" cy="30" r="10" stroke="black" stroke-width="2" fill="purple"></circle>
        <circle cx="200" cy="100" r="10" stroke="black" stroke-width="2" fill="purple"></circle>
        <circle cx="200" cy="170" r="10" stroke="black" stroke-width="2" fill="purple"></circle>
        <circle cx="200" cy="240" r="10" stroke="black" stroke-width="2" fill="purple"></circle>

	<!-- Draw activation layer -->
        <circle cx="250" cy="30" r="10" stroke="black" stroke-width="2" fill="blue"></circle>
        <circle cx="250" cy="100" r="10" stroke="black" stroke-width="2" fill="blue"></circle>
        <circle cx="250" cy="170" r="10" stroke="black" stroke-width="2" fill="blue"></circle>
        <circle cx="250" cy="240" r="10" stroke="black" stroke-width="2" fill="blue"></circle>

        <!-- Draw pre-output layer -->
        <circle cx="400" cy="100" r="10" stroke="black" stroke-width="2" fill="pink"></circle>
        <circle cx="400" cy="200" r="10" stroke="black" stroke-width="2" fill="pink"></circle>


                <!-- Draw arrows -->
        <!-- Connect input layer to hidden layer -->
        <!-- Input 1 to Hidden -->
        <line x1="60" y1="50" x2="190" y2="30" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="50" x2="190" y2="100" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="50" x2="190" y2="170" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="50" x2="190" y2="240" style="stroke:black;stroke-width:2"></line>

        <!-- Input 2 to Hidden -->
        <line x1="60" y1="150" x2="190" y2="30" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="150" x2="190" y2="100" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="150" x2="190" y2="170" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="150" x2="190" y2="240" style="stroke:black;stroke-width:2"></line>

        <!-- Input 3 to Hidden -->
        <line x1="60" y1="250" x2="190" y2="30" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="250" x2="190" y2="100" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="250" x2="190" y2="170" style="stroke:black;stroke-width:2"></line>
        <line x1="60" y1="250" x2="190" y2="240" style="stroke:black;stroke-width:2"></line>

	<!-- Connect hidden layer to activation layer (horizontal lines) -->
        <line x1="210" y1="30" x2="240" y2="30" style="stroke:black;stroke-width:2"></line>
        <line x1="210" y1="100" x2="240" y2="100" style="stroke:black;stroke-width:2"></line>
        <line x1="210" y1="170" x2="240" y2="170" style="stroke:black;stroke-width:2"></line>
	<line x1="210" y1="240" x2="240" y2="240" style="stroke:black;stroke-width:2"></line>
	
	<!-- Connect activation layer to output layer -->
	<line x1="260" y1="30" x2="390" y2="100" style="stroke:black;stroke-width:2"></line>
	<line x1="260" y1="30" x2="390" y2="200" style="stroke:black;stroke-width:2"></line>

	<line x1="260" y1="100" x2="390" y2="100" style="stroke:black;stroke-width:2"></line>
	<line x1="260" y1="100" x2="390" y2="200" style="stroke:black;stroke-width:2"></line>

	<line x1="260" y1="170" x2="390" y2="100" style="stroke:black;stroke-width:2"></line>
	<line x1="260" y1="170" x2="390" y2="200" style="stroke:black;stroke-width:2"></line>

	<line x1="260" y1="240" x2="390" y2="100" style="stroke:black;stroke-width:2"></line>
	<line x1="260" y1="240" x2="390" y2="200" style="stroke:black;stroke-width:2"></line>

<!-- Draw Softmax layer -->
<circle cx="450" cy="100" r="10" stroke="black" stroke-width="2" fill="red"></circle>
<circle cx="450" cy="200" r="10" stroke="black" stroke-width="2" fill="red"></circle>

<!-- Connect output layer to Softmax layer (horizontal lines) -->
<line x1="410" y1="100" x2="440" y2="100" style="stroke:black;stroke-width:2"></line>
<line x1="410" y1="200" x2="440" y2="200" style="stroke:black;stroke-width:2"></line>


	</svg>

Keeping this image in your head is extremely useful when writing the code for a neural network. This is what we're going to create, a fully-connected feedforward neural network (also traditionally called a multi-layer perceptron or MLP). Each circle is a neuron. Each neuron holds a single value (7, 0.01, -2.3, etc.). We call each vertical collection of neurons a *layer*. I make each layer here a different color. The crucial concept of a neural net is that it is made up of building blocks that transform the neuron values in one layer into different neuron values in the next layer, ultimately mapping the original input into some final output. Each neuron in one layer is connected to every other neuron in an adjacent layer by a *weight* (hence "fully connected"), a parameter that can be tuned to affect the value of the neuron it is connected to in the next layer. This collection of weights from every node in the left layer to every node in the right layer can be combined into a single *weight matrix* $W$. So whenever you see the criss-crossing of weight connections between two layers, that is one big weight matrix.

The green layer is the input layer, where each input neuron is some "feature" of a single training example. The purple is the pre-activation layer, the blue is the activation layer, the pink is what I'll call the pre-output layer, and the red is the softmax output layer. Softmax is a special math function that comes in handy when the goal of a network is classification (which of $n$ categories does the input belong to?), because it forces the output to be a proper probability distribution (all between 0 and 1, and all sum to 1).

The first building block of our network is the "linear" layer. It maps all the values in the input layer to different values in the pre-activation layer through matrix-vector multiplication with the matrix W. If we zoom into a single neuron-neuron pair between the input and pre-activation layers, the way we get the pre-activation neuron value is by multiplying the green neuron with the value of its weight which connects to the purple neuron, and then adding an extra term we call a bias. Collecting all these little multiplications and adds into one linear algebra operation, we get the expression for the linear layer $Wx + b$ where W is the weight matrix, $x$ is the input vector, and $b$ is a bias term (there's one bias per neuron in the subsequent layer)$. The bias term simply allows a weight of 0 to not force the subsequent neuron value to be 0 too.

The blue layer is the activation layer, and the way we get from purple to blue is by applying an *activation function*, or a *nonlinearity* (interchangeable because the activation function *must be nonlinear*. Most typical neural network diagrams ignore showing the pre-activation layer, because the application of an activation function is assumed to happen after a linear layer. In this case, since we're coding it from scratch, it is necessary to visualize every single operation the network is performing and visualizing it this way will especially come in handy when implementing backpropagation, the algorithm that allows the network to learn. Activation functions are very simple nonlinear functions, such as ReLU = $max(0,x)$, which transforms the input to 0 if it's negative, and otherwise keeps it the same. We'll use ReLU in our network.

![ReLU](/assets/relu.png){: width="500"}

The pink pre-output layer is just another linear layer involving the neuron values of the activation layer and a second weight matrix. So the same type of computation that got us from green to purple, gets us from blue to pink, just with a different input vector and a different weight matrix. You can think of the purple and pink neurons as being "raw", they are what you get directly out of the linear layer before an extra function is applied (either an activation function or a softmax for the final output).

The red output layer is the result of applying softmax to the entire pre-output layer. As an example, softmax$([1,3,2]) = [0.09003057, 0.66524096, 0.24472847]$. The real reason for using softmax, on top of the fact that it results in a probability distribution, is that whatever element was largest before, remains largest after, and the same for the second largest, and so on. Now we can interpret this to mean that the network predicts 66% that the input training example belongs to class 2, 9% for class 1, 24% for class 3. We could decide that whichever class has the highest probability becomes the network's class choice. This probability distribution will also come in handy when we use cross-entropy loss to get a notion of how badly the network is performing, and allow the weights to be updated in a manner so that the network does better next time.

The parameters of the network are the collection of weights and biases. Backpropagation is the algorithm that calculates the updates that need to be made to each parameter. The updates are determined via a process called gradient descent, where we take the partial derivative of the loss function with respect to each parameter, and then update the weight in the direction of the negative gradient, which ideally would make the loss decrease.

# Part 2. The Code

{% highlight python %}
import numpy as np
{% endhighlight %}

Trusty numpy for linear algebra operations and distributions for weight initialization!

{% highlight python %}
class ClassificationNeuralNetwork:
    def __init__(self, layer_sizes, LR=0.01, activation="relu", batch_size=64):
        self.layer_sizes = layer_sizes
        self.len_layer_sizes = len(layer_sizes)
        self.LR = LR
        self.batch_size = batch_size

        if activation == "relu":
            self.activation_fn = self.relu
            self.activation_fn_prime = self.relu_prime
            GAIN = 2**0.5
        elif activation == "sigmoid":
            self.activation_fn = self.sigmoid
            self.activation_fn_prime = self.sigmoid_prime
            GAIN = 1
        else:
            raise NotImplementedError("activation function provided is not implemented")

        # initialize parameters
        self.weights = []
        self.biases = []

        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * GAIN /(layer_sizes[i-1])**0.5)
            self.biases.append(np.zeros(layer_sizes[i]))
{% endhighlight %}

We make a class for the neural net and write its constructor. The first constructor argument is layer_sizes, which will be a list like [784,64,10] which means the input layer is size 784, the hidden layer is size 64 (we now refer to the "activation layer" in the last section as "hidden layer" here), and the output size is 10 (10 possible classes). Next we have LR, or the learning rate, which affects how big of an update each parameter gets per training iteration, and finally the batch size which will allow us to only iterate over small batches of a dataset at a time, getting a slightly noisy estimate of the gradients, and turning a gradient descent process into what's known as stochastic gradient descent. We then check which activation function the user specified, only concerning ourselves with two popular ones, ReLU and sigmoid (sigmoid is an S-curve with tails going to 0 in the negative direction and 1 in the positive direction, definitely nonlinear). Gain is a fairly advanced consideration, it is essentially a smart way of initializing the weights according to published research; even PyTorch uses this method by default. The gain is different depending on which activation function is being used. Then we initialize all the weights to be drawn from a normal distribution with standard deviation $\frac{\text{gain}}{\sqrt{\text{fan_in}}}$, where fan_in refers to the number of neurons in the previous layer (drawing from a standard normal distribution and then multiplying by a factor is a method of setting the standard deviation to be equal to that factor). We initialize the biases to 0.
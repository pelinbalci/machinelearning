#### Gradients

Gradients are important. 

- In general we need ⭐ DIFFERENTIABLE FUNCTIONS to be able to learn with neural networks.

#### Problems with Gradients 

-  Vanishing Gradients: If network gets too deep, (signal to noise ratio get bad) learning can become ⚡ SLOW.
The gradients for the lower layers (closer to the input) can become very small. In deep networks, computing these 
gradients can involve taking the product of many small terms.When the gradients vanish toward 0 for the lower layers, 
these layers train very slowly, or not at all.

Solution --> ⭐ReLu can be useful here and there are some other strategies, for example you can ⭐limit the depth⭐ of your model or 
find the minimum effective depth. 


- Exploding Gradients: If the weights in a network are very large, then the gradients for the lower layers involve products of many large 
terms. In this case you can have exploding gradients: gradients that get too large to converge.

Solution --> ⭐Lower the learning rate or ⭐batch normalization can help.


- ReLu layers can ⚡ DIE. =( 

Solution -->  It is possible since we have a hard cap at 0. If everything is below 0, there's no way for gradients to get propagated 
back through. Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. 
It outputs 0 activation, contributing nothing to the network's output, and gradients can no longer flow through it 
during backpropagation.  KEEP CALM and ⭐LOWER LEARNING RATES. 


#### Scaling

- Roughly zero-centered, [-1, 1] range often works well
- Helps gradient descent converge; avoid NaN trap
- Helps speed the converging. ⭐
- ⭐Avoiding outlier values can also help


#### Dropout Regularization

With probability P we take a node and remove it from the network for a single gradient step.
On different gradient steps we repeat and we'll take DIFFERENT nodes to drop out randomly.

(It works by randomly "dropping out" unit activations in a network for a single gradient step)

⭐ THE MORE you drop out, THE MORE regularization you have.

⭐ It gives strong results.  

- 0.0 = No dropout regularization.
- 1.0 = Drop out everything. The model learns nothing.
- Values between 0.0 and 1.0 = More useful.
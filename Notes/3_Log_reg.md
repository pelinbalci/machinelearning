#### Logistic Regression

For sklearn application: https://towardsdatascience.com/real-world-implementation-of-logistic-regression-5136cefb8125

It converts linear model raw prediction into the value between 0 -1, using sigmoid function.

Logistic regression allow our values to be interpreted as 0 to 1 and never exceeds that range. 

We can use the probability for calculating expected outcome. 

ex: P(house will sell) * price = expected outcome


#### Sigmoid Function 
Sigmoid Function gives us the value btw 0-1:
y_pred = 1 / (1 + e^ -(w.T * x + b))


#### Loss Function 
We can't use squared loss here. Instead:

log loss = &Sum; -y log(y_pred) - (1-y) log(1- y_pred)

This loss function looks like Shannon's Entropy Measure from information theory.

#### Regularization
Regularization is super important here. The loss function keeps trying to decrease loss to be 0 in higher dimension. 

We can use:

- L2 regularization --> penalize huge weights. 
- early stopping, limit training steps or learning rate

#### Why do we love log reg?
- If we need a method that scales well to MASSIVE data or
- If we need to use for extremely low latency predictions ---> we can use linear log reg. 


What is we need nonlinearities? We get them by adding feature cross product. 

#### Outcome of Log Reg

- As is 
- Convert to binary category: two mutually exclusive class. You can use treshold value. 
Treshold value can be tuned and it is very important. 


ex: If it returns 0.995, we can sure that the mail is spam. 
If it returns 0.003, it is not spam. 

But what if the outcome is 0.6 ? 
It is better to select a treshold. 

- Multiclass classification problem


#### Example

z = b + w1*x1 + w2*x2 + w3*x3  ----> linear regression 

z = 1 + 2*0 + (-1)*10 + 5*2

z = 1  ---> prediction of linear regression 


⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

Probability of class1 = p

z (prediction of linear regression ) is logodds. 

️logodds = log(p / 1-p)

z = log(p / 1-p)

p / 1 - p = e^z

p = e^z *(1- p)

p = e^z - e^z * p

p ( 1+ e^z) = e^z

p = e^z / 1 + e^z

p = 1 / 1 + e ^ -z

⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

Since z = 1: 

p = 1 / 1 + e ^ -1

p = 0.73

For the outcome '1', the probability is 0.73. This outcome shows that this point belong to class 1. 


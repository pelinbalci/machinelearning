#### Important!!!

Some Effective ML Guidelines
- Keep the first model simple
- Focus on ensuring data pipeline correctness
- Use a simple, observable metric for training & evaluation
- Own and monitor your input features
- Treat your model configuration as code: review it, check it in
- Write down the results of all experiments, especially "failures"


#### Machine learning Algorithms 

Machine learning Algorithms find a model which minimizes the loss. 

💛 Minimizing the loss ---> Empirical Risk Minimization

💛 Loss ---> penalty for a bad prediction. 

In linear regression we use L2 loss --> Squared Loss. (observed - prediction)^2

Mean Square Error =  1/N &Sum; (y - pred(X))^2 for X, y &isin; D

    where; D --> dataset
           X --> features
           y --> observed data
           N --> number of examples in D
           
           
           
#### How do we reduce loss?

https://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables


Derivative of (y - y_pred)^2 with respect to the weights and biases tells us how loss changes for a given x. 

Take small 📌steps in that direction. This strategy is called Gradient Descent. The steps 📌 are called Gradient Steps. 

* The plots of LOSS vs WEIGHTS is always CONVEX`. It has one minimum. 
* Regression problems yield a convex `loss vs weight plot. 
* Calculating the loss for every w is inefficient. Better mechanism is Gradient Descent. 


#### Partial Derivatives

Replies this question: how much a function changes when you change one variable a bit?

ex: 

    f(x,y) = e^2y sin(x)
    
 - &part;f(x,y) / &part;x = e^2y cos(x)
- &part;f(x,y) / &part;y = 2e^2y sin(x)

&nabla;f is a vector of partial derivatives with respect to all of the independent variables.


&nabla;f(x,y) = (&part;f(x)/&part;x ,  &part;f(x,y)/&part;y)

#### Gradient Descent

[Machine Learning Tutorial Python - 4: Gradient Descent and Cost Function](https://www.youtube.com/watch?v=vsWrXfO3wWw)

![image_description](/Users/pelin.balci/PycharmProjects/machinelearning/Notes/gradient_1.png)

![image_description](/Users/pelin.balci/PycharmProjects/machinelearning/Notes/gradient_2.png)


* Gradient ALWAYS points in the direction of great increase in LOSS function.
* In order to reduce loss, the gradient descent algorithm takes a step IN THE DIRECTION OF NEGATIVE GRADIENT.
* To determine the nex point in the loss function, the algorithm, adds 🧡some fraction🧡 of the gradient's magnitude to 
📌 the starting point.

🧡some fraction🧡 --> learning rate or step size. 
* Finding the perfect learning rate is NOT ESSENTIAL to successful model training. 
* SMALL learning rate needs a lot of computations. 
* The loss can be bigger than the previous one if the learning rate is too HIGH. 


📌 the starting point --> It matters where we start. 
* For convex problems, weights can start anywhere
* If there is more than one minimum & nonconvex function, strong dependency on initial values.

Computing gradient on a small value works well. 
Stochastic Gradient Descent --> one random sample for each itearion
Minibatch  Gradient Descent --> batches over 10 -1000
You iterate until overall loss stops changing at least changes extremely slow. When that happens THE MODEL HAS CONVERGED.


#### Hyperparameter vs Parameter

* Hyperparameter: You tweak during successive runs of training model. ex: Learning Rate
* Parameter: Weights are parameters whose values that machine learning system gradually learns. 


#### Generalization

ML is interested in generalization. 
The goal is predict well on unseen data. 
Model should be simple. 

3 basic assumptions:
- We draw examples 🔍i.i.d at random from the distribution.
- The distribution is 🔍stationary. It doesn't change over time. 
- We always pull from the same distribution. 


🔍i.i.d can be violated. --> a model that chooses ads to display. It can be affected from the ad user had previously seen. 
🔍stationary can be violated. --> retail sales info can be affected from seasonality. 


⭐️ PROBLEM:
We train on the training data, evaluate on test data and teh using the evaluation results, make changes in 
hyperparameters (learning rate or features). What is the problem here? 

Doing many rounds on this procedure may cause to implicitly fit the pecularities of spesific test set. 

⭐️ SOLUTION:

    Train model on training data --> evaluate model on VALIDATION SET --> tweak model according to the results ---> train model
    
    pick a model that does best on validation set --> confirm results on test set. 


❓Tweak Model : adjust anything; change learning rate, add- remove features, design a new model from scratch.


#### Representation

⭐️ Process of creating features from raw data --> FEATURE ENGINEERING


⭐️ One-hot-econding

ex: street name.

There may be a hundred different street names. We can use ONE-HOT-ENCODING and create a new vector:
[0, 0, 0, 0, 1, 0, 0, 0]

Don't use LABEL ENCODING for a street name. 

- house price = [100, 150, 140]
- room = [3, 4, 2]
- street = [main, xx, main]
- label_encoded_street = [1, 2, 1]

house_price = w1* room + w2*label_encoded_street

let's w1= 1 and w2 = 6

for xx--> 6*2 = 12 --> IT DOESN'T MEAN ANYTHING. 


⭐️ What is a good feature?

- It shouldn't be 0 in a lot of times. 
- It shouldn't be unique. (Never ever use ID)
- It shouşd be clear obvious meaning.
- It shouldn't have magic values. ( -1 may represent that 'the house has neer shown before'. Use 0 instead.)
- It shoulnd't change over time. (STATIONARITY)
- It shouldn't have outliers. 
- It shouldn't have missing values. Ex: use 'other' for categorical values. use 'mean or median' inorder not to effect the average value.

⭐️ Binning Trick

Create several boolean bins. ✔

️⭐️ Dense & Sparse Representation

Dense:

Word = [q, e, r, t, rt, rty, fg, fg, ad, qwe, b, vc, gh]
Occurence = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

Sparse:

word = [q, qwe]
Occurence = [1,1]

#### Scaling

Converting floating point feature values to their NATURAL RANGE.

- It helps gradient descent converge more quickly.
- It helps avoid NaN trap. 
- It helps model learn appropriate weights for each feature.

You DON'T HAVE TO GIVE every floating point EXACT SAME SCALE.

- standart scaling = value - mean/ std
- minmax scaling = [min_value, max_value] 

#### Outliers

- Take log. It will minimize the effect of outliers.
- Clip the values with a value. 

example: clip value with 4. min(feature, 4) We don't delete the values, but the values bigger than 4 is 4 now.

#### Remove

- ommited values
- duplicated values
- bad labels
- bad feature values

#### Feature Crosses

y = &sigma; (b + w1*x1 + w2*x2) ---> this is a linear model. 

We can produce a synthetic feature: x3 = x1 * x2

y = &sigma; (b + w1*x1 + w2*x2 + w3*x3) ---> we use a nonlinear feature in linear model. 

ex: 

binned_latitude = [0, 0, 1, 0]
binned_longitude = [1, 0, 0, 0]

binned_latitude * binned_longitude = [0, 1], [0,0], [0,0], [0,0],
                                     [0, 1], [0,0], [0,0], [0,0],
                                     [1, 1], [1,0], [1,0], [1,0],
                                     [0, 1], [0,0], [0,0], [0,0] 
                                   
This feature cross is a 16-element one-hot vector (24 zeroes and 1 one)













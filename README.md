# Digit-Recognization-using-MLP-and-CNN

Datset in this link : https://www.kaggle.com/datasets/crawford/emnist/data?select=emnist-balanced-train.csv

This repository explores two powerful deep learning architectures for digit prediction: Convolutional Neural Networks (CNNs) and Multilayer Perceptrons (MLPs).
**Data Preprocessing: **
After the data is converted to arrays, we divide each value by 255 to
normalize it. Afterward, we convert each value to "float32." This consistent scaling ensures that the pixel values across images are resized to a range of 0 to 1, which is necessary for neural networks to effectively assess the input data during training and inference stages. After that, the data is reorganized such that each image appears as a linear series of pixels, making it easier for the MLP to understand and facilitating effective pattern assimilation for accurate predictions. The training labels were converted into binary vectors using one-hot encoding, which ensures
that each class label is adequately represented for improved neural network training on classification tasks.
Furthermore, especially when working with images, we reformatted the input to fit the
precise format needed by Convolutional Neural Networks (CNNs). We ensure that the data is
correctly organized when we reshape it before training the CNN. We assist the CNN in
processing and learning from the photos by converting the data into a multidimensional array
with dimensions for width, height, and channels (such as color channels for RGB images or
grayscale images).

**Multi-Perceptron Architecture:** 
Our project uses the build_model_mlp function to create a
Multi-Layer Perceptron (MLP). The MLP begins with an input layer that has been designed to
process images with 28 x 28 pixel flats, giving us 784 features to work with.
Our approach incorporates hidden layers, where learning is aided by densely coupled
layers. By adjusting these layers, we modified the number of units by adjusting the units
hyperparameter, which has a range of 32 to 512 in multiples of 32. Furthermore, we have
experimented with several activation functions such as 'tanh', 'elu' , 'leaky_relu', each of which
could have a distinct effect on the model's performance.

**CNN Architecture:**
We built a Convolutional Neural Network (CNN) model using the Keras
framework for image classification tasks. Our model has several important components:
Convolutional layers that learn to extract useful features from the input images. We used
activation functions like ReLU, Leaky ReLU, ELU, or TANH to introduce nonlinearity. Batch
normalizationlayerswereaddedtostabilize andspeedupthetrainingprocess.Then,
Max-poolinglayers were added t oreduc ethe size of the feature maps and preven toverfitting.
Additional convolutional layers to extract more complicated features. Afterwards, Flatten layers
were added to convert the multi-dimensional feature maps into a format suitable for the dense 
(fully connected) layers.
Then, Dropout regularization were added in the dense layers to reduce overfitting. And finally a
softmax-activated layer for multi-class classification. We carefully optimized the model's
hyperparameters and evaluated its performance on validation and test datasets, measuring
metrics like accuracy, loss, confusion matrix, and classification report.
Hyperparameters used (MLP,CNN): We have carefully developed a methodical strategy
using Random Search CV to maximize the performance of our Convolutional Neural Network
(CNN) and Multilayer Perceptron (MLP) models. With the help of this method, we can
effectively explore the large hyperparameter space and find configurations that maximize
validation accuracy while consuming the least amount of computational power.
The build_model function, which contains important hyperparameters like the number of
filters, kernel sizes, activation functions, dropout rates, batch normalization usage, and
regularization strengths, optimizers, L1 and L2 regularizers and learning rate schedulers is the foundation of our CNN model. These hyperparameters affect CNN's capacity to extract pertinent features and generalize effectively to previously unseen data by defining its architecture and optimization tactics. Similar to this, the build_model_mlp function for the MLP model specifies variables such batch normalization, activation functions, dropout rates, and the number of units in dense layers.
We have set up our tuners with Random Search CV to perform a thorough investigation
of the hyperparameter space. In each trial, relevant models are built, hyperparameter combinations are sampled at random, and cross-validation techniques are used to assess the models' performance. A maximum of ten hyperparameter combinations per trial are taken into consideration in the goal of maximizing validation accuracy. To further improve the optimization process, we have included callbacks for learning rate scheduling during training.

_a) Optimizers:_ 
The optimizer weuse in our neural network training is essential to obtaining effective convergence and model performance. We've used a variety of optimizers designed for particular tasks: Adagrad adjusts learning rates to the size of gradients, balancing updates for frequent and infrequent parameters, whereas RMSprop adjusts the learning rate based on previous squared gradients to provide smoother updates. By dynamically altering learning rates without requiring initial settings, Adadelta further improves adaptability. Stochastic updates based on sampling gradients are the basis provided by SGD, which strikes a balance between noise robustness and efficiency.
Adam uses momentum in conjunction with RMSprop's adaptive learning rates to provide
quicker convergence and better generalization. Adam combines the benefits of momentum and adaptive learning rates from RMSprop, potentially leading to quicker
convergence and better generalization in our dataset.
_b) Adaptive learning rate:_ 
We've employed the Step decay and Cosine annealing decay
methods as part of our adaptive learning rate strategy. The two techniques systematically reducethelearningrate at specifi cpoints in the training process.Cosineannealingdecay adjusts the learning rate followin ga cosinefunction's decay pattern.Thi senhances the _training dynamics and helps the model converge more effectively in our dataset.
c) Batch Normalization:_
It improves the training dynamics of neural network models by stabilizing and accelerating convergence through the normalization of layer activations
within mini-batches. We could see that better results were obtained with
Batch_Normalization ON.
_d) Dropout Layer:_
One of the key strategies to prevent overfitting in a neuralnetwork is to incorporate dropout layers, which randomly deactivate a percentage of neurons during training. This encourages the network to focus on more diverse features. By adjusting dropout rates such as 0.2 and 0.4, we can strike a good balance between preventing overfitting while still preserving important information, ultimately improving the model's
ability to generalize..
_e) L1 and L2 regularization:_ 
L1 regularization encourages sparsity by penalizing the absolute values of weights, while L2 regularization promotes smoother weight distributions in our dataset by penalizing the squared values, both enhancing the model's generalization performance by constraining weight magnitudes.
_f) Activation Function:_ 
Neural networks depend on activation functions tohelp the model recognize intricate patterns in the data. ReLU has the "dying ReLU" issue, yet it offers effective training by directly outputting the input for positive values. This problem is solved by Leaky ReLU, which permits a modest gradient for negative inputs. Because elu maintains non-zero outputs for negative inputs, it allows for faster convergence. Tanh is zero-centered and sigmoid and tanh are appropriate for binary classification applications. When using Softmax for multi-class classification, the output probabilities
are guaranteed to add up to one. Every activation function offers distinct benefits and is selected according to the network architecture and task specifications.

**Overfitting/Underfitting:**
We didn’t come across Overfitting/Underfitting since the training
accuracy and testing accuracy were quite good. But if in case, we came across overfitting we
would have ensured that Dropout-Layers, better schedulers and Flattening Layers were used,
since they help in introducing non-linearity into the model and thus helping the model learn
complex patterns. As for underfitting, I would have ensured that we have trained with enough
amount of data and would have also looked into whether the learning rate is between 0.01 -
0.001 . Otherwise, there are chances it won’t converge and cause underfitting.
Pros/Cons: MLPs can handle varying input sizes but struggle to capture the spatial relationships
in images, limiting their effectiveness for tasks like character recognition. On the other hand,
CNNs excel at image-related tasks due to their ability to learn hierarchical features through
convolutional and pooling layers. This allows CNNs to effectively capture important patterns in
EMNIST characters.

![image](https://github.com/arjunk3x/Digit-Recognization-using-MLP-and-CNN/assets/147756161/6d51c3e0-4bdf-4f8f-9cc3-290f8bb46670)

![image](https://github.com/arjunk3x/Digit-Recognization-using-MLP-and-CNN/assets/147756161/162a7626-23d2-4972-b1c7-144c59ac828d)

![image](https://github.com/arjunk3x/Digit-Recognization-using-MLP-and-CNN/assets/147756161/07b54f69-ef16-4ef5-930c-540629e9567d)

![image](https://github.com/arjunk3x/Digit-Recognization-using-MLP-and-CNN/assets/147756161/f12c7b27-7112-4d60-b639-56f07d9f5455)



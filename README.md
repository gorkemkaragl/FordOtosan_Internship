
# PROJECT QUESTIONS

### What is machine learning?
Machine learning is an application of artificial intelligence that uses statistical techniques to enable computers to learn and make decisions without being explicitly programmed. It is predicated on the notion that computers can learn from data, spot patterns, and make judgments with little assistance from humans.

It is a subset of Artificial Intelligence. It is the study of making machines more human-like in their behavior and decisions by giving them the ability to learn and develop their own programs. This is done with minimum human intervention, i.e., no explicit programming. The learning process is automated and improved based on the experiences of the machines throughout the process.

### What is Unsupervised vs Supervised learning difference?
Supervised learning is a learning algorithm that helps predict unseen data based on training data. It uses labeled training datasets to achieve desired outcomes.

Unsupervised learning, on the other hand, is a machine learning technique where users do not need to supervise the model. Instead, it allows the model to work on its own to discover previously unidentified patterns and information. It primarily deals with unlabeled data.

The main difference is that in supervised learning, the model learns based on inputs and specific outputs, whereas in unsupervised learning, the model attempts to discover nonlinear relationships and structures within the data.

### What is Deep Learning?
Deep learning is a machine learning method consisting of multiple layers that predicts outcomes based on a given dataset. It involves training computer systems using algorithms called artificial neural networks to perform data analysis, recognition, and understanding tasks. The deep learning method relies on using multi-layered artificial neural networks to discover and learn complex structures and patterns within the data.

### What is Neural Network(NN)?
A neural network is a type of machine learning algorithm modeled based on the structure and functions of the human brain. It is designed to recognize patterns and make predictions or decisions without being explicitly programmed. Neural networks consist of interconnected layers of "neurons" that process and transmit information. They are commonly used in various fields such as image and speech recognition, natural language processing, and other applications.

### What is Convolution Neural Network (CNN)?
Convolutional Neural Network (CNN) is a type of artificial neural network in the field of deep learning, particularly used for processing and understanding spatial data such as images and audio.

#### Advantages:
- When CNNs (Convolutional Neural Networks) process data, they share weights considering that similar features can be found frequently in different parts of the input. This reduces the number of parameters and helps the network effectively learn features.

- In spatial data such as images, relationships between elements are crucial. CNNs (Convolutional Neural Networks) learn spatial dependencies by using filters and convolution operations on input data to recognize spatial structures. As a result, they achieve better and more effective results.

### What is segmentation task in NN? Is it supervised or unsupervised?
Segmentation, fundamentally, refers to the process of dividing a given input data into different parts or regions. This division aims to highlight specific patterns or features within the data. For instance, segmentation can be used to recognize different objects or structures within an image. This process finds widespread application in various fields, such as object detection, separating organs in medical images, and enabling autonomous vehicles to detect roads and objects.

The task of segmentation can be approached through both supervised and unsupervised methods.

### What is classification task in NN? Is it supervised or unsupervised?
Classification task is an important type of Artificial Neural Networks (ANNs), and its main purpose is to categorize input data into different classes or categories. Classification is typically evaluated under supervised learning, a machine learning approach where a model learns from labeled examples in the training data. Thus, the classification task is a type of supervised learning where the model is trained using both input data and their corresponding correct outputs (labels).

The classification task finds numerous real-world applications. For example, it is widely used in areas such as email spam filtering, disease diagnosis, handwriting recognition, and image recognition.

### Compare segmentation and classification in NN.
- Classification is the task of using an artificial neural network to categorize input data into different categories. On the other hand, Segmentation is the task of using an artificial neural network to divide input data into smaller parts or sections.

- Classification is the process of assigning given inputs to specific classes or categories, limiting each input to a single class. Segmentation, on the other hand, aims to divide a large data piece, such as an image, into smaller parts like pixels, regions, or objects.

- Classification is generally considered under supervised learning, which requires training data to be labeled (with correct class labels). Segmentation, on the other hand, can be addressed using unsupervised, semi-supervised, or supervised learning methods. It can be a fully supervised approach, which requires labeled data, or an unsupervised approach where unlabeled data can be used.

### What is data and dataset difference?
- Data can essentially be thought of as raw information and data pieces. A dataset, on the other hand, is a collection of multiple data items brought together.

- Data can be expressed in various forms such as numbers, text, audio, images, or other formats. A dataset is a structured group of similar types of data elements, prepared to be used for specific analysis, modeling, or processing tasks.

- Data can be obtained from various sources and can be used for all kinds of processing or analysis. A dataset includes data that can be utilized for training, validating, and testing machine learning models.

### What is the difference between supervised and unsupervised learning in terms of dataset?
In supervised learning, the model utilizes a labeled dataset to learn the relationship between input and output and make predictions accordingly. On the other hand, in unsupervised learning, the model attempts to discover the structure and relationships within the unlabeled dataset, identifying clusters and patterns within the data.

# DATA PREPROCESSİNG
## Extracting Masks

### What is color space ?
Color space is a mathematical model used to define, represent, and process colors in various applications such as visual processing, graphic design, color editing, and many other fields where managing and communicating colors is essential.

Color spaces have a structure that defines the coordinates of colors within a color model. These coordinates allow us to represent colors with numerical values. There are different types of color spaces, each with different use cases and properties.

### What RGB stands for ? 
RGB stands for "Red, Green, Blue." It consists of three primary colors that form the basis of colored images and screens. The combination of these colors allows the creation of different colors. By merging these three colors with varying brightness values and combinations, the entire range of colors can be generated.

### In Python, can we transform from one color space to another?
Yes, it is possible to perform color conversions in the Python language. There are various libraries and tools available for color conversions in Python. Particularly, popular image processing libraries like "OpenCV" and "Pillow" can be used for performing color space transformations. These libraries allow you to convert colors between different color spaces efficiently.

### What is the popular library for image processing?
The most popular and widely used library for image processing is known as "OpenCV" (Open Source Computer Vision Library) in the Python language. OpenCV is an open-source image processing and computer vision library.

## Converting into Tensor
### Explain Computational Graph.
Computational graphs are a type of graph that can be used to represent mathematical expressions. This is similar to descriptive language in the case of deep learning models, providing a functional description of the required computation.
In general, the computational graph is a directed graph that is used for expressing and evaluating mathematical expressions. 

Here are a few basic terminologies in computational graphics

- A variable is represented by a node in a graph. It could be a scalar, vector, matrix, tensor, or even another type of variable.
- A function argument and data dependency are both represented by an edge. These are similar to node pointers.
- A simple function of one or more variables is called an operation. There is a set of operations that are permitted. Functions that are more complex than these operations in this set can be represented by combining multiple operations.

### What is Tensor ?
Tensors are simply mathematical objects that can be used to describe physical properties, just like scalars and vectors. In fact tensors are merely a generalisation of scalars and vectors; a scalar is a zero rank tensor, and a vector is a first rank tensor.

### What is one hot encoding?
Categorical data refers to variables that are made up of label values, for example, a “color” variable could have the values “red,” “blue,” and “green.” Think of values like different categories that sometimes have a natural ordering to them.

### What is CUDA programming?
CUDA programming is a parallel computing platform and programming model developed by NVIDIA for utilizing the power of NVIDIA GPUs (Graphics Processing Units) to perform general-purpose computation tasks.

# Neural_Network
  
  
  ## What is an Artificial Neural Network ?

  Artificial Neural Network is used in a particuliar domain of the machine learning : <b>the deep learning</b>.
  It's concept is actually inspired by the human brain, by simply replacing the synapses and neurons by numbers.

  The network is composed of layers, themselves composed of neurons.
  Each neurons has a threshold to be activated, it is represented by a value called the <b>bias</b>. 
  The network is composed by at least two layers, called respectively <b>input layer</b> and <b>output layer</b>.
  If there is more than these two layers, they should be between the both and are called "hidden layers"
  The layers are linked by what we called <b>weights</b>. A given neuron in a layer is linked with each neurons in the previous layer.
  So that each neurons in a given layer affect all the following layers neurons depending of the weights impact.
     
  <img src="https://miro.medium.com/max/791/0*hzIQ5Fs-g8iBpVWq.jpg" width="50%" height="50%">

---

 ## How works a Neural Network ?

  By processing data and the targeted output, the Neural Network use <b>stochastic gradient descent</b> to reduce how wrong the neural network's output is between its current     result and the targeted value. So that it can reduce the error at each new steps.

  -It works in two different ways :
  
  <h3>The forward propagation :</h3>
  <img src="https://images.deepai.org/django-summernote/2019-06-06/5c17d9c2-0ad4-474c-be8d-d6ae9b094e74.png">
  It assign the given input to the input layer neuron's values, then compute the next layer using the biases and the weights linked between themselves and the latter         neuron's. Keep repeating the operation for each hiddens layers until it reach the output layer and give the corresponding outpout.
       
  <h3>The back propagation :</h3>
  <img src="https://www.guru99.com/images/1/030819_0937_BackPropaga1.png" width="50%" height="50%">
  It compute the loss of the last layer, then propagate it using weights and biases to get the gradient of each layers so that we can use it for the gradient descent latter on.

---

 ## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)

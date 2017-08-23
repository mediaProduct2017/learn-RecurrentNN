# learn-RecurrentNN

## 1. Recurrent neural net的结构

与basic neural net相比，每一层Recurrent neural net有额外的输入，也有额外的输出。

![Recurrent neural net](images/RNN.png)

![Recurrent neural net II](images/rnn2.png)

一般的，h(t-1)是输入，h(t)是输出，在这里，额外的输入是x(t-1)，额外的输出是y(t-1).

## 2. The probability of a sequence of words

预测The probability of a sequence of words，在两个方面有用，一是自动翻译方面，不同的语言词序不同，二是语音识别方面，同音字到底选取哪个要看整个句子。

![markov model](images/markov.png)

The probability of a sequence of words转化为已知一组单词，预测下一个单词是某个具体单词的概率。

RAM requirement scales with number of words. 预测下一个单词时，一组单词的所有信息都包含在之前一个neuron当中，所以，内存只需要保存之前一个neuron的信息，所需的内存有限。

## 3. Comparison with other neural networks

可以认为，RecurrentNN是一种实用的Tree recursive neural network，包含着specific assumption。

对于RecurrentNN，可能没有额外的输出y(t-1)，但是在结构上，都是每次加入一个新的neuron。在自然语言处理方面，就是每次加入一个word，或者每次加入一个字。

## 4. References

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 5. Function and application

1. RecurrentNN常用来产生text，也就是用机器来自动写文章，自动翻译
2. 也可以用机器来自动生产图片（模仿其他图片，避开侵权），比如使用最新的deep learing技术Generative adversarial networks (GANs)

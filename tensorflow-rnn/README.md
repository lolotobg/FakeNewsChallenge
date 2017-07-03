## Code is here:  
https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb

## Tutorial is here:
https://www.tensorflow.org/tutorials/recurrent#recurrent-neural-networks

## Data set:   
PTB dataset from Tomas Mikolov's webpage

http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

https://catalog.ldc.upenn.edu/ldc99t42

## To install tensorflow on Windows:
- Install python 3.5.*, 64bit version
- Make sure you have the latest version of pip:<br/>
  _$pip install --upgrade pip_<br/>
- Install tensorflow<br/>
  _$pip install --upgrade tensorflow_<br/>
  
  
 ## Run the tutorial:
  
  `tar xvfz simple-examples.tgz -C $HOME`
  
  `cd models/tutorials/rnn/ptb`
  `python ptb_word_lm.py --data_path=$HOME/simple-examples/data/ --model=small`
  
  There are 3 supported model configurations in the tutorial code: "small", "medium" and "large". The difference between them is in size of the LSTMs and the set of hyperparameters used for training.
  
  
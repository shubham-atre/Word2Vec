# Word2Vec
In this project I have implemented methods to learn word representations building on Skip-grams and applied the learned word representations to a semantic task.

# Implementation details:

* word2vec_basic.py : 

Contains implementation to generate a Word2Vec model using batch training and applying Gradient Descent Optimizer on Cross Entropy and Noise Contrastive Estimation Functions.

* loss_func.py :

Conatains Implementation of the two loss funtions cross entropy and NCE, using tensorflow.

* word_analogy.py :

This script reads an input file line by line, and
Iterates over the examples and finds the average of the differences between their vectors, and
Iterates over the choices and finds those pairs, whose vector differences have least and most cosine similarity with average of the examples.

* score_maxdiff.pl :
  
  A perl script to evaluate our predictions on development data
    Usage:</br>
      ./score_maxdiff.pl word_analogy_mturk_answers.txt _your prediction file_ _output file of result_
	  
	 Example:</br>
	  ./score_maxdiff.pl word_analogy_mturk_answers.txt Output_word_analogy_out_cross_entropy.txt Accuracy_word_analogy_out_cross_entropy.txt
	  
* Results :

Note: All the experiments were performed in Google Collab, using Python2.

• Configurations for best NCE model:</br>
batch_size,	num_skips,	skip_window,	max_num_steps,	Avg Loss,	Accuracy</br>
260	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	13	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		7	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		250001	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		1.28	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;34.9

• Configurations for  best Cross Entropy model:</br>
batch_size,	num_skips,	skip_window,	max_num_steps,	Avg Loss,	Accuracy</br>
128			&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	8		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	4		&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;	200001	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;		4.82	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;33.9

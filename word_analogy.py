import os
import pickle
import numpy as np

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

input_filepath = './word_analogy_test.txt';

file = open(input_filepath, 'r')

line = file.readline()

output_file = open("./word_analogy_test_predictions_nce.txt","w+")

output_line = ''

#Reading the file line by line
while line:
    eg_choices = line.split('||')
    eg = eg_choices[0].split(',')
    choices = eg_choices[1].replace('\n', '').split(',')
    output_line = output_line + eg_choices[1].replace(',', ' ').replace('\n', '')

    sumo_diff = 0
    numo_eg = 0
    avg_diff = 0

    #Iterating over the examples and finding the average of the differences between their vectors
    for e in eg:
        e_split = e.strip('\"').split(':')
        eg_left = embeddings[dictionary[e_split[0]]]
        eg_right = embeddings[dictionary[e_split[1]]]
        eg_diff = np.subtract(eg_left,eg_right)
        sumo_diff =sumo_diff + eg_diff
        numo_eg = numo_eg +1

    avg_diff = np.divide(sumo_diff, numo_eg)
    min_similarity = float("inf")
    max_similarity = float("-inf")

    '''
    Iterating over the choices and finding those pairs, 
    whose vector differences have least and most cosine similarity with average of the examples
    '''
    for c in choices:
        c_split = c.strip('\"').split(':')
        c_left = embeddings[dictionary[c_split[0]]]
        c_right = embeddings[dictionary[c_split[1]]]
        c_diff = np.subtract(c_left, c_right)

        cosine_sim = np.dot(avg_diff, c_diff) / (np.sqrt(np.dot(avg_diff,avg_diff)) * np.sqrt(np.dot(c_diff,c_diff)))

        if cosine_sim<min_similarity:
            min_similarity = cosine_sim
            min_pair = c

        if cosine_sim>max_similarity:
            max_similarity = cosine_sim
            max_pair = c

    output_line = output_line + " " + min_pair + " " + max_pair

    output_file.write(output_line)

    output_line = '\n'

    line = file.readline()

output_file.close()
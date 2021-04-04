from keras.utils import to_categorical
import json
import dynet as dy
import random
import math
import numpy as np
from numpy import argmax
from itertools import compress
from numpy.random import multinomial


#glove_dir = 'glove.6B.50d.txt'
glove_dir = "glove.6B.100d.txt"
json_dir = 'unim_poem.json'
data_len = 1000

def read_json():
	with open(json_dir, "r") as file:
	    items_dict =  json.load(file)
	data_len = len(items_dict)
	items_dict = items_dict[0:data_len]

	bigram_model = []    
	whole_lines = []

	for i in range(len(items_dict)):
		for poem_line in items_dict[i]["poem"].split("\n"):
			whole_lines.append(poem_line)

			poem_line = poem_line.split(" ")
			len_poem_line = len(poem_line)
			bigram_model.append(('<s>' , poem_line[0]))

			for p in range(len_poem_line-1):

				bigram_word = poem_line[p] + " " + poem_line[p+1]
				bigram_word_tuple = (poem_line[p] , poem_line[p+1])
				if (bigram_word != " "):
					bigram_model.append(bigram_word_tuple)

				if( p == len_poem_line-2 ):
					bigram_model.append((poem_line[p+1] ,  '</s>'))
				else:
					bigram_model.append((poem_line[len_poem_line - 1] ,  '\n'))

	return whole_lines, bigram_model

def make_bigram_indexes(bigram_model, word_idxs):
	bigram_indexes = []
	for b in bigram_model:
		if (b[0] != '' and b[0] != None  and b[1] != '' and b[1] != None):
			bigram_indexes.append((word_idxs[b[0]], word_idxs[b[1]]))

	return bigram_indexes


def make_word_embedings(unigram_model):
	file = open(glove_dir, encoding="utf8")
	embeddings = {}
	for line in file.readlines():
	    line = line.split()
	    key = line[0]
	    value = []
	    for i in line[1:]:
	    	value.append(float(i))

	    embeddings[key] = value


	word_embeddings = []
	prev =""
	for uni in unigram_model:
		if uni in embeddings:
			word_embeddings.append(embeddings[uni])
			prev = uni
		elif (prev != ''):
			word_embeddings.append(embeddings[prev])

	for i in range(3):
		prev = unigram_model[-i-1]
		word_embeddings.append(embeddings[prev])

	return embeddings, np.array(word_embeddings)


def make_unigram_model(whole_lines):
	unigram_model = []

	start_words = ['<s>', '</s>', '\n']
	unigram_model.extend(start_words)
	

	for l in range(len(whole_lines)):
		for w in whole_lines[l].split():
			if(w != None):
				unigram_model.append(w)
	

	embeddings, word_embeddings = make_word_embedings(unigram_model)


	idx = [i for i in range(len(unigram_model))]
	for i in range(len(unigram_model)):
		idx.append(i)
	onehot_vect = to_categorical(idx)

	word_idxs = {unigram_model[argmax(onehot_vect[i])] : i for i in range(len(unigram_model))}


	return unigram_model, onehot_vect, word_idxs, embeddings, word_embeddings



def train_and_generate_poem(bigram_indexes, unigram_model,  word_embeddings, num_epochs, onehot_vect, word_idxs):
	dynet_model = dy.Model()

	x_dim = word_embeddings.shape[0]
	y_dim = word_embeddings.shape[1]


	param_a = dynet_model.add_lookup_parameters((y_dim,x_dim), init=word_embeddings)
	param_b = dynet_model.add_parameters(y_dim)
	param_c = dynet_model.add_parameters((x_dim, y_dim))
	param_d = dynet_model.add_parameters(x_dim)

	trainer = dy.SimpleSGDTrainer(dynet_model)

	print("x_dim = ", x_dim)
	print("y_dim = ", y_dim)

	W = dy.parameter(param_a)
	b = dy.parameter(param_b)
	U = dy.parameter(param_c)
	d = dy.parameter(param_d)


	for epoch in range(num_epochs):
		epoch_loss = 0.0
		for word1 , word2  in bigram_indexes[:data_len-1]:
			dy.renew_cg()


			x_val = dy.inputVector(list(onehot_vect[word1]))
			h_val = dy.tanh(W * x_val + b)
			y_val = U * h_val + d


			loss = dy.pickneglogsoftmax(y_val, word2)
			epoch_loss += loss.scalar_value()

			loss.backward()
			trainer.update()

		print('Epoch', epoch, '- Loss =', epoch_loss/x_dim)


	current = '<s>'
	poem = ''
	probabilities = []

	number_of_word_in_line = 0

	for i in range(40):
		dy.renew_cg()

		x_val = dy.inputVector(list(onehot_vect[word_idxs[current]]))
		h_val = dy.tanh(W * x_val + b)
		y_val = U * h_val + d

		probs = dy.softmax(y_val)

		if len(current) > 2 and current!='</s>' and current!='\n':
		    poem += (current + " ")
		    number_of_word_in_line += 1   

		if number_of_word_in_line == 6 :
		    poem += '\n'
		    number_of_word_in_line = 0

		    

		current = next(compress(unigram_model, multinomial(1, probs.value(), 1)[0]))
		probabilities.append(probs.__getitem__(word_idxs[current]).value())

	return poem, probabilities 


def calculate_perplexity(probabilities):
    total_probability = 0
    for probability in probabilities:
        total_probability += math.log2(probability)

    return 1 / math.pow(2, (total_probability/len(probabilities)))

def main():
	whole_lines, bigram_model = read_json()
	print("~~~~~~ BIGRAM IS GENERATED ~~~~~~")
	print(bigram_model[:50])
	print('Length Bigram Model= {}'.format(len(bigram_model)))

	unigram_model, onehot_vect, word_idxs, embeddings, word_embeddings = make_unigram_model(whole_lines)
	print("~~~~~~ UNIGRAM IS GENERATED ~~~~~~")
	print(unigram_model[:50])
	print('Length Unigram Model= {}'.format(len(unigram_model)))


	print("~~~~~~ WORD EMBEDDINGS IS GENERATED ~~~~~~")
	print("Embeddings for word poultry")
	print(embeddings["poultry"])

	bigram_indexes = make_bigram_indexes(bigram_model, word_idxs)

	poem, probabilities = train_and_generate_poem(bigram_indexes, unigram_model, word_embeddings,1, onehot_vect, word_idxs)

	print("My Poem...\n", poem)
	poem_perplexity = calculate_perplexity(probabilities)
	print("Perplexity = ",poem_perplexity)



if __name__ == "__main__":
    main()

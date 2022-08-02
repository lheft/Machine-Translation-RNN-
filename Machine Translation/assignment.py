import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: French train data (all data for training) of shape (num_sentences, window_size)
	:param train_english: English train data (all data for training) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""
	chop=train_english[:,:-1]
	label=train_english[:,1:]
	batches=len(train_french)/model.batch_size
	batches=int(batches)
	
	for i in range(batches):
		mbs=model.batch_size
		enc_i=train_french[i*mbs:(i+1)*mbs]
		dec_i=chop[i*mbs:(i+1)*mbs]
		batch_label=label[i*mbs:(i+1)*mbs]
		mask=np.where(batch_label!=eng_padding_index,1,0)
		
		with tf.GradientTape() as tape:
			logit=model.call(enc_i,dec_i)
			loss=model.loss_function(logit,batch_label,mask)
			p=tf.cast(np.sum(mask),dtype=tf.float32)
			accuracy=model.accuracy_function(logit,batch_label,mask)
			pacc=p*accuracy
			if i %100:
					print(loss/pacc)
		
		grad=tape.gradient(loss,model.trainable_variables)

		model.optimizer.apply_gradients(zip(grad,model.trainable_variables))


			


	# NOTE: For each training step, you should pass in the French sentences to be used by the encoder,
	# and English sentences to be used by the decoder
	# - The English sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

	pass

@av.test_func
def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: French test data (all data for testing) of shape (num_sentences, window_size)
	:param test_english: English test data (all data for testing) of shape (num_sentences, window_size + 1)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set,
	e.g. (my_perplexity, my_accuracy)
	"""
	chop=test_english[:,:-1]
	label=test_english[:,1:]
	batches=len(test_french)/model.batch_size
	batches=int(batches)
	total_loss=0
	total_correct=0
	pol=0

	for i in range(batches):
		mbs=model.batch_size
		e_input=test_french[i*mbs:(i+1)*mbs]
		d_input=chop[i*mbs:(i+1)*mbs]
		batch_label=label[i*mbs:(i+1)*mbs]
		mask=np.where(batch_label!= eng_padding_index,1,0)
		pol1=tf.cast(np.sum(mask), dtype=tf.float32)
		pol+=pol1

		logi=model.call(e_input,d_input)
		loss=model.loss_function(logi,batch_label,mask)
		total_loss+=loss
		accuracy=model.accuracy_function(logi,batch_label,mask)
		total_correct+= accuracy* pol1
	per_sym_acc=total_correct/pol

	perplexity=tf.exp(total_loss/pol)	

	return perplexity, per_sym_acc	

	# Note: Follow the same procedure as in train() to construct batches of data!

	pass

def main():
	
	model_types = {"RNN" : RNN_Seq2Seq, "TRANSFORMER" : Transformer_Seq2Seq}
	if len(sys.argv) != 2 or sys.argv[1] not in model_types.keys():
		print("USAGE: python assignment.py <Model Type>")
		print("<Model Type>: [RNN/TRANSFORMER]")
		exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=False)

	print("Running preprocessing...")
	data_dir   = '../../data'
	file_names = ('fls.txt', 'els.txt', 'flt.txt', 'elt.txt')
	file_paths = [f'{data_dir}/{fname}' for fname in file_names]
	train_eng,test_eng, train_frn,test_frn, vocab_eng,vocab_frn,eng_padding_index = get_data(*file_paths)
	print("Preprocessing complete.")

	model = model_types[sys.argv[1]](FRENCH_WINDOW_SIZE, len(vocab_frn), ENGLISH_WINDOW_SIZE, len(vocab_eng))

	# TODO:
	# Train and Test Model for 1 epoch.


	train_eng,test_eng,train_frn,test_frn,vocab_eng,vocab_frn,eng_padding_index

	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	model_args=(FRENCH_WINDOW_SIZE,len(vocab_frn),ENGLISH_WINDOW_SIZE,len(vocab_eng))
	if sys.argv[1]=='RNN':
			model=RNN_Seq2Seq(*model_args)
	elif sys.argv[1]=='TRASNFORMER':
			model=Transformer_Seq2Seq(*model_args)

	train(model,train_frn,train_eng,eng_padding_index)
	t,v=test(model,test_frn,test_eng,eng_padding_index)
	print(t,v)
	av.show_atten_heatmap()
	pass

if __name__ == '__main__':
	main()

import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the French vocab
		self.english_vocab_size = english_vocab_size # The size of the English vocab

		self.french_window_size = french_window_size # The French window size
		self.english_window_size = english_window_size # The English window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 125
		self.optimizer=tf.keras.optimizers.Adam(learning_rate=.001)
		stddev=.1

		self.E=tf.Variable(tf.random.normal(shape=[self.english_vocab_size,self.embedding_size],stddev=stddev))
		self.F=tf.Variable(tf.random.normal(shape=[self.french_vocab_size,self.embedding_size],stddev=stddev))

		self.positional_enc=transformer.Position_Encoding_Layer(self.french_window_size,self.embedding_size)

		self.enc=transformer.Transformer_Block(self.embedding_size,is_decoder=False,multi_headed=False)
		self.dec=transformer.Transformer_Block(self.embedding_size,is_decoder=True,multi_headed=False)
		
		self.dense=tf.keras.layers.Dense(self.english_vocab_size,activation='softmax')


		# Define English and French embedding layers:

		# Create positional encoder layers

		# Define encoder and decoder layers:

		# Define dense layer(s)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to French sentences
		:param decoder_input: batched ids corresponding to English sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		fembedded=tf.nn.embedding_lookup(self.F,encoder_input)
		fpos_enc=self.positional_enc(fembedded)

		encoded=self.enc(fpos_enc)

		Eenbedded=tf.nn.embedding_lookup(self.E,decoder_input)
		Epos_enc=self.positional_enc(Eenbedded)

		decoded=self.dec(Epos_enc,context=encoded)

		prob=self.dense(decoded)
		# TODO:
		#1) Add the positional embeddings to French sentence embeddings
		#2) Pass the French sentence embeddings to the encoder
		#3) Add positional embeddings to the English sentence embeddings
		#4) Pass the English embeddings and output of your encoder, to the decoder
		#5) Apply dense layer(s) to the decoder out to generate probabilities

		return prob

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		loss=tf.keras.losses.sparse_categorical_crossentropy(labels,prbs)
		maskedL=tf.boolean_mask(loss,mask)
		return tf.reduce_sum(maskedL)

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)

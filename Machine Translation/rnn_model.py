from base64 import encode
import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the French vocab
		self.english_vocab_size = english_vocab_size # The size of the English vocab

		self.french_window_size = french_window_size # The French window size
		self.english_window_size = english_window_size # The English window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters

		batch_size=300
		embedding_size=125
		LR=.001
		stddev=.1
		activation='softmax'
		units=200

		self.batch_size = batch_size # You can change this
		self.embedding_size = embedding_size # You should change this
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
		
		self.E = tf.Variable(tf.random.normal(shape=[self.english_vocab_size, self.embedding_size], stddev=stddev))
		self.F = tf.Variable(tf.random.normal(shape=[self.french_vocab_size, self.embedding_size], stddev=stddev))
		
		self.encoder = tf.keras.layers.GRU(units)
		self.decoder = tf.keras.layers.GRU(units, return_sequences=True)
		self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation=activation)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to French sentences
		:param decoder_input: batched ids corresponding to English sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		F_embedding=tf.nn.embedding_lookup(self.F,encoder_input)
		encoder=self.encoder(F_embedding)

		enbedding=tf.nn.embedding_lookup(self.E,decoder_input)
		decoded=self.decoder(enbedding, initial_state=encoder)

		logits=self.dense(decoded)


		return logits

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
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		loss = tf.keras.losses.sparse_categorical_crossentropy(labels,prbs)
		bool_mask=tf.boolean_mask(loss,mask)
		
		return tf.reduce_sum(bool_mask)

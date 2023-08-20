import data_loading_helpers as dlh
from configuration import configs
import tensorflow as tf
from keras import layers
import keras

# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super().__init__()
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)
#
#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)
#
#
# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super().__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
#
#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions

import numpy as np


def positional_encoding(max_length, d_model):
    angle_rads = np.arange(max_length)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] / d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sine function to even indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cosine function to odd indices
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding


class MultiHeadAttention:
    def __init__(self, num_heads):
        self.num_heads = num_heads

    def __call__(self, inputs, mask=None):
        input_dim = inputs.get_shape()[-1]
        head_dim = input_dim // self.num_heads

        queries = tf.layers.dense(inputs, input_dim, activation=tf.nn.relu, name="queries")
        keys = tf.layers.dense(inputs, input_dim, activation=tf.nn.relu, name="keys")
        values = tf.layers.dense(inputs, input_dim, activation=tf.nn.relu, name="values")

        queries = tf.concat(tf.split(queries, self.num_heads, axis=-1), axis=0)
        keys = tf.concat(tf.split(keys, self.num_heads, axis=-1), axis=0)
        values = tf.concat(tf.split(values, self.num_heads, axis=-1), axis=0)

        attention_scores = tf.matmul(queries, keys, transpose_b=True)
        if mask is not None:
            attention_scores += (1.0 - mask) * -1e9
        scaled_attention_scores = attention_scores / tf.sqrt(tf.cast(head_dim, tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)

        weighted_values = tf.matmul(attention_weights, values)
        weighted_values = tf.concat(tf.split(weighted_values, self.num_heads, axis=0), axis=-1)
        output = tf.layers.dense(weighted_values, input_dim, activation=tf.nn.relu, name="output")

        return output


class PositionwiseFeedForward:
    def __init__(self, num_units, hidden_units):
        self.num_units = num_units
        self.hidden_units = hidden_units

    def __call__(self, inputs):
        hidden_output = tf.layers.dense(inputs, self.hidden_units, activation=tf.nn.relu, name="hidden")
        output = tf.layers.dense(hidden_output, self.num_units, activation=None, name="output")
        return output


class PositionalEncodingLayer:
    def __init__(self, max_length, d_model):
        self.max_length = max_length
        self.d_model = d_model
        self.pos_encoding = tf.constant(positional_encoding(max_length, d_model), dtype=tf.float32)

    def __call__(self, inputs):
        batch_size, seq_length, _ = tf.unstack(tf.shape(inputs))
        pos_encodings = tf.slice(self.pos_encoding, [0, 0, 0], [batch_size, seq_length, self.d_model])
        return inputs + pos_encodings


def dataset_construction():
    complete_config = configs.complete_config(configs.default_config)
    db = dlh.new_data_box(complete_config)
    seed = 113  # configs.get("Random_Seed", 113)
    db.shuffle_data(seed)
    db.oversample_underreepresented_classes()
    return complete_config, db


class AugmentedTransformer:

    def __init__(self, input_config={}, vocab_size=None, max_sentence_length=41):
        self.input_config = configs.complete_config(input_config)
        self.data_placeholders = {}
        self.sentence_features = []
        self.we_dim = 300
        self.embedding_dim=128
        self.et_features_size = 5
        self.eeg_features_size = 105
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.maxlen = self.max_sentence_length
        self.lstm_units = self.input_config['LSTM_UNITS']
        self.num_classes = 2 if self.input_config['BINARY_CLASSIFICATION'] else 3
        #self.sentence_level_features()
        # self.model = self.create_model()
        self.CreateGraph()

    # def sentence_level_features(self):
    #     if self.input_config['WORD_EMBEDDINGS']:
    #         self._create_word_embedding_variables()
    #     if self.input_config['EYE_TRACKING']:
    #         self._create_et_variables()
    #     if self.input_config['EEG_SIGNAL']:
    #         self._create_eeg_variables()
    #
    #     self.sentence_features = tf.concat(self.sentence_features, 2)

    def _create_word_embedding_variables(self):
        self.data_placeholders['WORD_IDXS'] = tf.placeholder(tf.int32, [None, self.max_sentence_length], name="input_x")
        self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.we_dim], -1.0, 1.0), name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.data_placeholders['WORD_IDXS'])
        self.sentence_features.append(self.embedded_chars)

    def _create_et_variables(self):
        self.data_placeholders['ET'] = tf.placeholder(tf.float32,
                                                      [None, self.max_sentence_length, self.et_features_size],
                                                      name="input_et")
        self.sentence_features.append(self.data_placeholders['ET'])

    def _create_eeg_variables(self):
        if self.input_config['EEG_TO_PIC']:
            self.data_placeholders['EEG'] = tf.placeholder(tf.float32
                                                           ,
                                                           [None, self.max_sentence_length, self.eeg_features_size, 1],
                                                           name="input_eeg")

        else:
            self.data_placeholders['EEG'] = tf.placeholder(tf.float32
                                                           , [None, self.max_sentence_length, self.eeg_features_size],
                                                           name="input_eeg")
            self.sentence_features.append(self.data_placeholders['EEG'])

    def CreateGraph(self):
        self.data_placeholders['SEQUENCE_LENGTHS'] = tf.placeholder(tf.int32, [None], name="sequence_lengths")
        self.data_placeholders['TARGETS'] = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")

        self.DROPOUT_KEEP_PROB = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.L2_REG_LAMBDA = tf.placeholder(tf.float32)
        self.L1_REG_LAMBDA = tf.placeholder(tf.float32)
        self.LEARNING_RATE = tf.placeholder(tf.float32)

        with tf.device(self.input_config['TF_DEVICE']), tf.name_scope("embedding"):

            # Build the input
            if self.input_config['WORD_EMBEDDINGS']:
                self._create_word_embedding_variables()
            if self.input_config['EYE_TRACKING']:
                self._create_et_variables()
            if self.input_config['EEG_SIGNAL']:
                self._create_eeg_variables()

            self.sentence_features = tf.concat(self.sentence_features, 2)

        with tf.device(self.input_config['TF_DEVICE']):
            self.transformer_input = self.sentence_features

            self.attention_output = MultiHeadAttention(num_heads=8)(self.transformer_input)
            self.feed_forward_output = PositionwiseFeedForward(num_units=self.embedding_dim,
                                                               hidden_units=self.embedding_dim * 4)(
                self.attention_output)

            positional_encoder = PositionalEncodingLayer(max_length=self.max_sentence_length,
                                                         d_model=self.embedding_dim)
            self.sentence_features_with_pos = positional_encoder(self.feed_forward_output)

            self.embedding = tf.reduce_mean(self.sentence_features_with_pos, axis=1)
            self.embedding_dim = self.embedding.get_shape()[1]

        if self.input_config['HIDDEN_LAYER_UNITS'] > 0:
            self.hidden = tf.layers.dense(self.embedding, units=self.input_config['HIDDEN_LAYER_UNITS'])
            self.scores = tf.layers.dense(self.hidden, units=self.num_classes)
        else:
            self.scores = tf.layers.dense(self.embedding, units=self.num_classes)

        self.probabilities = tf.nn.softmax(self.scores, 1, name="probabilities")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

        self.actual_values = tf.argmax(self.data_placeholders['TARGETS'], 1, name="true_values")

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.data_placeholders['TARGETS'])
        vars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars])
        l1_loss = tf.add_n([tf.norm(v, ord=1) for v in vars])
        self.classification_loss = tf.reduce_mean(losses) + self.L2_REG_LAMBDA * l2_loss + self.L1_REG_LAMBDA * l1_loss
        self.loss = 0.0

        loss_summary = tf.summary.scalar("loss", self.classification_loss)

        classification_summaries = add_tf_classification_measures(self)

        self.summary = tf.summary.merge([classification_summaries, loss_summary])

    # def create_model(self):
    #     inputs = layers.Input(shape=(self.maxlen,))
    #     embedding_layer = TokenAndPositionEmbedding(self.maxlen, vocab_size, embed_dim)
    #     x = embedding_layer(inputs)
    #     transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    #     x = transformer_block(x)
    #     x = layers.GlobalAveragePooling1D()(x)
    #     x = layers.Dropout(0.1)(x)
    #     x = layers.Dense(20, activation="relu")(x)
    #     x = layers.Dropout(0.1)(x)
    #     outputs = layers.Dense(2, activation="softmax")(x)
    #     model = keras.Model(inputs=inputs, outputs=outputs)
    #
    #     return model


# def train_transformer(self):
#     self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#     history = self.model.fit(
#         x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
#     )


def main():
    complete_config, dataset = dataset_construction()
    model = AugmentedTransformer(input_config=complete_config, vocab_size=len(dataset.vocab_processor.vocabulary_),
                                 max_sentence_length=dataset.vocab_processor.max_document_length)


if __name__ == '__main__':
    main()

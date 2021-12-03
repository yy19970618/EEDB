import tensorflow as tf
import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers import Layer, Input, Embedding, LSTM, Dense, MultiHeadAttention
from tensorflow.keras.models import Model
# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, -1), tf.float32)

#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class Agent:
    def get_model1(self,hidden_units,k):
        encoder_inputs = Input(shape=(None,61), name="encode_input")
        enc_outputs, enc_state_h, enc_state_c =  LSTM(16, return_sequences=True, return_state=True)(encoder_inputs)
        attn_output, _ = MultiHeadAttention(num_heads=2, key_dim=16)(enc_outputs[:, k:,:],enc_outputs,return_attention_scores=True)
        decoder_out, _, _ = LSTM(63, return_sequences=True, return_state=True)(attn_output)
        model = Model(inputs=[encoder_inputs], outputs=[decoder_out])
        model.summary()
        return model
    def get_model(self,hidden_units,k):
        model =  keras.Sequential([
            Input(shape=(None,61), name="encode_input"),
            tf.keras.layers.Masking(mask_value=[-1.]*61, input_shape=(None, 61)),
            self.get_model1(hidden_units,k)
            ])
        model.summary()
        return model
t=tf.range(10)
tf.print(t)
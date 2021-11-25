import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,RNN,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.optimizers import Adam,Nadam


tokenizer_enc = pickle.load(open(r"/content/grammar_correction/app/tokenizer_enc_20211112.pkl", "rb"))
tokenizer_dec = pickle.load(open(r"/content/grammar_correction/app/tokenizer_dec_20211112.pkl", "rb"))

class Encoder(tf.keras.layers.Layer):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,inp_vocab_size,embedding_size,enc_units,input_length):
        
        super().__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(input_dim=inp_vocab_size, output_dim=300, input_length=input_length,
                           mask_zero=True,name="embedding_layer_encoder")
        self.lstmcell = tf.keras.layers.LSTMCell(enc_units)
        self.enc = RNN(self.lstmcell,return_sequences=True, return_state=True)

    def call(self,input_sequence,states):

        embedding_enc = self.embedding(input_sequence)
        enc_output, enc_state_h, enc_state_c = self.enc(embedding_enc, initial_state=states)
        return enc_output, enc_state_h, enc_state_c

    def initialize_states(self,batch_size):

        ini_hidden_state = tf.zeros([batch_size,self.enc_units])
        ini_cell_state = tf.zeros([batch_size,self.enc_units])
        return [ini_hidden_state,ini_cell_state]

class Attention(tf.keras.layers.Layer):

  '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
  '''
  def __init__(self,att_units):
    super().__init__()
    self.softmax = Softmax(axis=1)

  def call(self,decoder_hidden_state,encoder_output):
    attention_weight = tf.matmul(encoder_output,tf.expand_dims(decoder_hidden_state,axis=2))
    context = tf.matmul(tf.transpose(encoder_output, perm=[0,2,1]),attention_weight)
    context = tf.squeeze(context,axis=2)
    output = self.softmax(attention_weight)
    return context,output
    
class OneStepDecoder(tf.keras.Model):
  def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units  ,att_units):
        super().__init__()
        self.tar_vocab_size = tar_vocab_size
        self.dec_units = dec_units
        self.att_units = att_units
        self.embedding = Embedding(input_dim=tar_vocab_size, output_dim=300, input_length=input_length,
                           mask_zero=True,name="embedding_layer_decoder") 
        self.lstmcell = tf.keras.layers.LSTMCell(dec_units)
        self.decoder_lstm = RNN(self.lstmcell,return_sequences=True, return_state=True)
        self.dense   = Dense(tar_vocab_size)
        self.attention=Attention(self.att_units)

  def call(self,input_to_decoder, encoder_output, state_h,state_c):
        embedding_layer = self.embedding(input_to_decoder)
        embedding_layer = tf.squeeze(embedding_layer,axis=1)
        context_vector,attention_weights=self.attention(state_h,encoder_output)
        context_vector_for_concat = tf.concat([context_vector,embedding_layer],1)
        context_vector_for_concat = tf.expand_dims(context_vector_for_concat,1)
        deco_output, deco_state_h, deco_state_c = self.decoder_lstm(context_vector_for_concat,initial_state=[state_h,state_c])
        output_after_dense_layer = self.dense(deco_output)
        output_after_dense_layer = tf.squeeze(output_after_dense_layer,axis=1)
        return output_after_dense_layer,deco_state_h, deco_state_c,attention_weights,context_vector

class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units  ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
        super().__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.att_units = att_units
        self.input_length = input_length
        self.onestepdecoder = OneStepDecoder(self.out_vocab_size,self.embedding_dim,self.input_length,self.dec_units,self.att_units)
        
    @tf.function    
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):
        all_outputs = tf.TensorArray(tf.float32,size=input_to_decoder.shape[1])
        for timestep in range(input_to_decoder.shape[1]):
            output,decoder_hidden_state,decoder_cell_state,attention_weights,context_vector=self.onestepdecoder(input_to_decoder[:,timestep:timestep+1],encoder_output,decoder_hidden_state,decoder_cell_state)
            all_outputs = all_outputs.write(timestep,output)
        all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
        #print("all outpt shape is ",all_outputs.shape)
        return all_outputs
    
class encoder_decoder(tf.keras.Model):
  def __init__(self,inp_vocab_size,out_vocab_size,embedding_size,lstm_size,input_length,batch_size,att_units,*args):
    super().__init__() # https://stackoverflow.com/a/27134600/4084039
    self.encoder = Encoder(inp_vocab_size,embedding_size,lstm_size,input_length)
    self.decoder = Decoder(out_vocab_size,embedding_size,input_length,lstm_size,att_units)
    self.batch = batch_size
  
  def call(self,data):
    input,output = data[0], data[1]
    l = self.encoder.initialize_states(self.batch)
    encoder_output,encoder_final_state_h,encoder_final_state_c = self.encoder(input,l)
    decoder_output = self.decoder(output,encoder_output,encoder_final_state_h,encoder_final_state_c)
    return decoder_output

enc_vocab_size = len(tokenizer_enc.word_index) + 1 
dec_vocab_size = len(tokenizer_dec.word_index) + 1 
embedding_dim=300
input_length=12
lstm_size=192
batch_size=512
att_units =192

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    """ Custom loss function that will not consider the loss for padded zeros.
    in this loss function we are ignoring the loss
    for the padded zeros. i.e when the input is zero then we do not need to worry what the output is. 
    This padded zeros are added from our end
    during preprocessing to make equal length for all the sentences.

    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
optimizer=Nadam(learning_rate=0.001)

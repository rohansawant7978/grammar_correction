import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,RNN,Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Softmax
from tensorflow.keras.optimizers import Adam,Nadam
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.translate.gleu_score import sentence_gleu
from utilities import *

model = encoder_decoder(enc_vocab_size,dec_vocab_size,embedding_dim,lstm_size,input_length,batch_size,att_units)
model.load_weights('/content/drive/MyDrive/Self Case studies/CS02 Grammar Error Corrector/Models/03 enc_dec_with_attention/03_enc_dec_with_attention')

def function1(input_sentence):
    '''This dunction takes sentence as input and returns a grammatically correct sentence as output'''
    corrc_wrd_idx_dict = tokenizer_dec.word_index
    corrc_idx_wrd_dict = {v: k for k, v in corrc_wrd_idx_dict.items()}

    input_sentence = tokenizer_enc.texts_to_sequences([input_sentence])[0]
    initial_hidden_state = tf.zeros([1,192])
    initial_cell_state = tf.zeros([1,192])
    encoder_initial_state = [initial_hidden_state,initial_cell_state]
    input_sentence = tf.keras.preprocessing.sequence.pad_sequences([input_sentence],maxlen=12,padding='post')
    input_sentence = input_sentence[0]
    enc_output, enc_state_h, enc_state_c = model.layers[0](np.expand_dims(input_sentence,0),encoder_initial_state)
    pred = []
    sentence = []
    cur_vec = np.ones((1, 1),dtype='int')
    attention_array = np.zeros([12,12])
    for i in range(12):
        output,dec_state_h, dec_state_c,att_weights,context_vector = model.layers[1].onestepdecoder(cur_vec, enc_output,enc_state_h, enc_state_c)
        enc_state_h, enc_state_c = dec_state_h, dec_state_c
        cur_vec = np.reshape(np.argmax(output), (1, 1))
        if corrc_idx_wrd_dict[cur_vec[0][0]] == '<end>':
            break
        pred.append(cur_vec[0][0])
        att_weights = tf.squeeze(att_weights)   
        attention_array[i] = att_weights
    for i in pred:
        sentence.append(corrc_idx_wrd_dict[i])
    return " ".join(sentence)

def function2(file):
    df = pd.read_csv(file)
    glue_score_arr = []
    for i in tqdm(range(len(df))):
        reference = [df['correct'].iloc[i].split()]
        pred = function1(df['incorrect'].iloc[i])
        candidate = pred.split()
        try:
            glue_score_arr.append(sentence_gleu(reference, candidate))
        except:
            continue
    return np.mean(glue_score_arr)

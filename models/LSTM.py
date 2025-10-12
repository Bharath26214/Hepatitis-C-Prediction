from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

class AttLSTM(Model):
    def __init__(self, input_shape, learning_rate=1e-4):
        super(AttLSTM, self).__init__()

        self.input_layer = Input(shape=input_shape, name='protein_input')

        self.encoder_lstm = LSTM(
            50, return_sequences=True, return_state=True, 
            activation='relu', kernel_initializer=glorot_uniform()
        )
        
        self.attention = Attention()
        self.concat = Concatenate(axis=-1)
        
        self.decoder_lstm = LSTM(
            50, return_sequences=False, activation='relu',
            kernel_initializer=glorot_uniform()
        )
        
        self.dropout = Dropout(0.1)
        self.fc1 = Dense(100, activation='relu', kernel_initializer=glorot_uniform(), bias_initializer='zeros')
        self.fc2 = Dense(50, activation='relu', kernel_initializer=glorot_uniform(), bias_initializer='zeros')
        
        self.output_layer = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(), bias_initializer='zeros')

        self.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def call(self, inputs):
        encoder_output, state_h, state_c = self.encoder_lstm(inputs)
        
        attention_output = self.attention([encoder_output, encoder_output])
        context_vector = self.concat([encoder_output, attention_output])
        
        decoder_output = self.decoder_lstm(context_vector)
        
        x = self.dropout(decoder_output)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return self.output_layer(x)
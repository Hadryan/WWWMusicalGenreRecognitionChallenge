from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling2D, concatenate
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam

class Network:
    def __init__(self, with_compile=True, num_tags=16):
        self.num_tags = num_tags
        self.regularizer = l2(1e-5)
        self.model = self.get_model()


    def get_model(self, with_compile=True):
        # input layer
        audio_input = Input(shape=(1536,), name='input')
        x = audio_input
        x = Dropout(0.5)(x)

		# hidden layer
        x = Dense(1024, kernel_regularizer=self.regularizer, name='dense')(x)
        x = ELU()(x)
        x = Dropout(0.5)(x)

		# output layer
        x = Dense(16, kernel_regularizer=self.regularizer, name='output')(x)
        x = Activation('softmax')(x)
		
		# model
        model = Model(audio_input, x)


        if with_compile:
            optimizer = Adam(lr = 0.001)
            model.compile(optimizer=optimizer,
                          loss = 'categorical_crossentropy',
                          metrics=['accuracy'])
        return model

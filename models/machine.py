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
        audio_input = Input(shape=(128, 43, 1), name='input')
        x = audio_input

		# Layer 1
        x = Conv2D(16, (5,5), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2,1), strides=(2,1), name='MP_1')(x)

		# Layer 2
        x = Conv2D(32, (3,3), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_2')(x)
        x = BatchNormalization(axis=3)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_2')(x)
        x = Dropout(0.1)(x)

		# Layer 3
        x = Conv2D(64, (3,3), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_3')(x)

		# Layer 4
        x = Conv2D(64, (3,3), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_4')(x)
        x = BatchNormalization(axis=3)(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_4')(x)
        x = Dropout(0.1)(x)

		# Layer 5
        x = Conv2D(128, (3,3), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_5')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='MP_5')(x)

		# Layer 6
        x = Conv2D(256, (3,3), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_6')(x)
        x = ELU()(x)

		# layer 7
        x = Conv2D(256, (1,1), padding='same', kernel_regularizer=self.regularizer,
				name = 'conv_7')(x)
        x = BatchNormalization(axis=3)(x)
        x = ELU()(x)

		# GAP
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)

		# Dense
        x = Dense(256, kernel_regularizer=self.regularizer, name = 'dense')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(0.5)(x)

		# output
        x = Dense(self.num_tags, kernel_regularizer=self.regularizer, name='output')(x)
        x = Activation('softmax')(x)

		# model
        model = Model(audio_input, x)


        if with_compile:
            optimizer = Adam(lr = 0.001)
            model.compile(optimizer=optimizer,
                          loss = 'categorical_crossentropy',
                          metrics=['accuracy'])
        return model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error


data_path = "./"
trval = pd.read_csv(data_path + 'train_sample.csv')
test = pd.read_csv(data_path + 'test_sample.csv')
feats = [f"feature_{i}" for i in range(0, 3)]

x_trval, y_trval = trval[feats], trval["target"]
x_train, x_val, y_train, y_val = train_test_split(x_trval, y_trval, test_size=0.20, random_state=4, stratify=y_trval)


x_train = [x_train[f].values for f in feats]
x_val = [x_val[f].values for f in feats]
x_test = [test[f].values for f in feats]

y_train = y_train.values
y_val = y_val.values


def get_embed(x_input, x_size, out_dim, embedding_reg=0.0002):
    # x_input is index of input (either user or item)
    # x_size is length of vocabulary (e.g. total number of users or items)
    # out_dim is size of embedding vectors
    if x_size > 0: #category
        embed = Embedding(x_size, out_dim, input_length=1, embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(out_dim, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed


def build_model(f_size, k_latent=2, kernel_reg=0.05):
    dim_input = len(f_size)
    input_x = [Input(shape=(1, )) for i in range(dim_input)] 
    lin_terms = [get_embed(x, size, 1) for (x, size) in zip(input_x, f_size)]
    factors = [get_embed(x, size, k_latent) for (x, size) in zip(input_x, f_size)]
    s = Add()(factors)
    diffs = [Subtract()([s, x]) for x in factors]
    dots = [Dot(axes=1)([d, x]) for d, x in zip(diffs, factors)]
    x = Concatenate()(lin_terms + dots)
    x = BatchNormalization()(x)
    output = Dense(1, activation='relu', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    model.compile(optimizer=Adam(clipnorm=0.6, learning_rate=0.001), loss='mean_squared_error')
    return model


n_epochs = 200
batch_size = 64

f_size = [int(x_trval[f].max()) + 1 for f in feats]
model = build_model(f_size)
earlystopper = EarlyStopping(patience=20, verbose=0, restore_best_weights=True)
model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=0, validation_data=(x_val, y_val), callbacks=[earlystopper])

p = np.squeeze(model.predict(x_val))
print("RMSE", mean_squared_error(y_val, p) ** 0.5)

preds_test = np.round(np.squeeze(model.predict(x_test))).astype(int)
test['target'] = preds_test
test.to_csv('submission.csv', index=False, columns=["ID", "target"])
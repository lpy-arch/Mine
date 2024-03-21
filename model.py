import keras

def build_model(output_units, num_units, dense_units, loss, learning_rate):
    
    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))

    x = input
    # num_units是一个列表，存储着多个LSTM单元的单元数，下面的代码代表连接多层分别以num_units中元素为单元数的LSTM层
    # keras中LSTM的输入为3维(batch, timesteps, feature)，默认输出为2维(batch, feature)
    # 若想多个LSTM层连在一起就需要让输出也是3维，设置return_sequences=True
    # 但全连接层需要2维输入，因此要在最后一层LSTM中设置return_sequences=False
    for i, units in enumerate(num_units):
        if i == len(num_units) - 1:
            x = keras.layers.LSTM(units, return_sequences=False)(x)
        else:
            x = keras.layers.LSTM(units, return_sequences=True)(x)

    x = keras.layers.Dropout(0.2)(x)

    # dense_units是一个列表，存储着多个全连接层的神经元数，下面的代码代表连接多层分别以dense_units中元素为神经元数的全连接层
    for units in dense_units:
        x = keras.layers.Dense(units, activation="softmax")(x)
    output = x

    model = keras.Model(input, output)
    
    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    
    model.summary()
    
    return model
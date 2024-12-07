import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,GlobalAveragePooling2D,BatchNormalization,Dropout
from tensorflow.keras import regularizers

from Utils.SA import SoftAttention
# from SA import SoftAttention


def ResNet50_SA_model():
    resnet = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

    # Exclude the last 3 layers of the model.
    conv = resnet.layers[-3].output
    print("Conv shape before SoftAttention ResNet50:", conv.shape)

    attention_layer,map2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
    print("Attention layer shape ResNet50:", attention_layer.shape)

    attention_layer=(MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    conv=(MaxPooling2D(pool_size=(2, 2),padding="same")(conv))

    print("Conv shape after pooling ResNet50:", conv.shape)
    print("Attention layer shape after pooling ResNet50:", attention_layer.shape)

    conv = concatenate([conv,attention_layer])
    conv  = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)

    output = GlobalAveragePooling2D()(conv)
    output = Dense(2048, activation='relu', kernel_regularizer=regularizers.l1(0.01))(output)
    output = BatchNormalization()(output)
    output = Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.01))(output)
    output = BatchNormalization()(output)
    output = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(output)
    output = BatchNormalization()(output)
    output = Dense(7, activation='softmax')(output)
    model = Model(inputs=resnet.input, outputs=output)
    print('++++++++++++++++++++++++++++++++')
    return model
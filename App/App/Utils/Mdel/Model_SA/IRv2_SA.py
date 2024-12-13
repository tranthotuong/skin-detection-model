import tensorflow as tf
import logging
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,BatchNormalization,Dropout
from tensorflow.keras import regularizers

from Utils.SA import SoftAttention
# from SA import SoftAttention

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def IRv2_SA_model():

    irv2 = tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax",

    )

      # TESST
    dummy_input_1 = tf.random.normal([1, 8, 8, 192])
    soft_attention = SoftAttention(aggregate=True, m=16, concat_with_x=False, ch=192,name='soft_attention')
    attention_layer_1, attention_maps_1 = soft_attention(dummy_input_1)

    logging.info(f"Attention maps shape: {attention_layer_1.shape}")

    # Excluding the last 28 layers of the model.
    conv = irv2.layers[-28].output
    logging.info(f"Conv shape before SoftAttention IRv2: {conv.shape}")

    ch=conv.shape[-1]
    attention_layer, _ = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(conv.shape[-1]),name='soft_attention')(conv)
    logging.info(f"Attention layer shape IRv2: {attention_layer.shape}")

    attention_layer=(MaxPooling2D(pool_size=(2, 2),padding="same")(attention_layer))
    conv=(MaxPooling2D(pool_size=(2, 2),padding="same")(conv))
    logging.info(f"Conv shape after pooling IRv2: {conv.shape}")
    logging.info(f"Attention layer shape after pooling IRv2: {attention_layer.shape}")


    conv = concatenate([conv, attention_layer])
    conv  = Activation('relu')(conv)
    conv = Dropout(0.5)(conv)


    output = Flatten()(conv)
    output = Dense(2048, activation='relu', kernel_regularizer=regularizers.l1(0.01))(output)
    output = BatchNormalization()(output)
    output = Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.01))(output)
    output = BatchNormalization()(output)
    output = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(output)
    output = BatchNormalization()(output)
    output = Dense(7, activation='softmax')(output)
    model = Model(inputs=irv2.input, outputs=output)
    return model
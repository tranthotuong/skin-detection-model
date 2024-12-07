from Utils import preprocess_image as IG
# import preprocess_image as IG

import cv2
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
import numpy as np
import streamlit as st


# Function to make predictions
def predict(image,models):
    img_224, img_299 = IG.resize_image(image)
    models_load = models
    targetnames_dict = {0:'akiec', 1:'bcc', 2:'bkl', 3:'df', 4:'mel',5: 'nv', 6:'vasc'}

    predictions = []
    score_dict={}
    for model,file in models_load.items():
        if str(model).__contains__("IRv2"):
            img_299= tf.keras.applications.inception_resnet_v2.preprocess_input(img_299)
            print('img', img_299.shape)
            prediction = file.predict(img_299)
            st.write("=================================")
            st.write("Model - ", model)
            for scores in [prediction[0]]:
                scores=list(scores)
                for i in range(len(scores)):
                    score_dict[i]=scores[i]

            for key, value in score_dict.items():
                for lable_code, targetname in targetnames_dict.items():
                    if key == lable_code:
                        st.write(f"Label: {targetname} - Score: {value:.2f}")
            predictions.append(prediction)
        else:
            img_224 = tf.keras.applications.inception_resnet_v2.preprocess_input(img_224)
            prediction = file.predict(img_224)
            st.write("=================================")
            st.write("Model - ", model)
            for scores in [prediction[0]]:
                scores=list(scores)
                for i in range(len(scores)):
                    score_dict[i]=scores[i]

            for key, value in score_dict.items():
                for lable_code, targetname in targetnames_dict.items():
                    if key == lable_code:
                        st.write(f"Label: {targetname} - Score: {value:.2f}")    
            predictions.append(prediction)
    
    
    # Hard Voting: Aggregate predictions from multiple models
    avg_predictions = np.mean(predictions, axis=0)
    final_prediction = np.argmax(avg_predictions)

    return final_prediction


import streamlit as st
from PIL import Image
import numpy as np
from Utils import predict
from Utils import models as Model


def main():
    st.title('Dermatology Diagnosis')
    st.sidebar.title('Upload')

    # Load the model
    try:
        st.sidebar.write("Loading model...")
        models = Model.load_model()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return

    # Sidebar options
    option = st.sidebar.selectbox('Choose an option', ('Upload Image',))

    if option == 'Upload Image':
        st.sidebar.write('Please upload an image in JPG, JPEG, or PNG format.')
        uploaded_file = st.sidebar.file_uploader(
            'Upload an image', 
            type=['jpg', 'jpeg', 'png'], 
            help='Accepted formats: JPG, JPEG, PNG'
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)

                if st.button('Diagnose'):
                    # Run prediction
                    prediction = predict.predict(image, models)

                    # Label decoding
                    label_decode = {
                        0: 'Actinic keratoses (akiec)',
                        1: 'Basal cell carcinoma (bcc)',
                        2: 'Benign keratosis-like lesions (bkl)',
                        3: 'Dermatofibroma (df)',
                        4: 'Melanoma (mel)',
                        5: 'Melanocytic nevi (nv)',
                        6: 'Vascular lesions (vasc)'
                    }

                    diagnosis = label_decode.get(prediction, "Unknown")
                    st.write("=================================")
                    st.write(f"Prediction: **{diagnosis}**")
            except Exception as e:
                st.error(f"Error during diagnosis: {e}")


if __name__ == '__main__':
    main()

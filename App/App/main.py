from PIL import Image, UnidentifiedImageError
import requests
import numpy as np
import sys
import json
from Utils import predict
from Utils import models as Model
import argparse

# Load the model only once
def load_model():
    try:
        return Model.load_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# Process the image
def process_image(image_url, models):
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Open the image
        image = Image.open(response.raw)

        # Convert image to numpy array for prediction
        # image_array = np.array(image).copy()  # Create a copy to avoid reference issues

        # Ensure the array is in the correct shape for the model
        # Example: Resize the image to (224, 224) for models like ResNet
        # resized_image_array = np.resize(image_array, (224, 224, 3))  # Adjust size and dimensions as required

        # Run prediction
        prediction, confidence, detailed_results  = predict.predict(image, models)

        # Label decoding
        # Label decoding with IDs
        label_decode = {
            0: {"id": 2, "name": "Actinic keratoses (akiec)"},
            1: {"id": 3, "name": "Basal cell carcinoma (bcc)"},
            2: {"id": 4, "name": "Benign keratosis-like lesions (bkl)"},
            3: {"id": 5, "name": "Dermatofibroma (df)"},
            4: {"id": 6, "name": "Melanoma (mel)"},
            5: {"id": 7, "name": "Melanocytic nevi (nv)"},
            6: {"id": 8, "name": "Vascular lesions (vasc)"},
        }
        diagnosis = label_decode.get(prediction, {"id": -1, "name": "Unknown"})

        return {
            "rep_skin_detection": {
                "data": {
                    "image_url": image_url,
                    "diagnosis": diagnosis,
                    "confidence": np.float32(confidence),
                    "detailed_results":detailed_results
                },
                "code": 200 
            }          
        }

    except UnidentifiedImageError:
        return {"error": "The provided URL does not contain a valid image"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch image from URL: {e}"}
    except Exception as e:
        return {"error": f"Error during diagnosis: {e}"}

# Custom function to handle numpy types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    if isinstance(obj, (np.float32, np.float64)):
        return round(float(obj), 2)  # Convert numpy float to native Python float
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert numpy int to native Python int
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Skin diagnosis script")
    parser.add_argument("--image_url", required=True, help="URL of the image to diagnose")
    args = parser.parse_args()

    # Extract the image URL
    image_url = args.image_url

    # Load the model
    try:
        models = load_model()
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    # Process the image and get diagnosis
    result = process_image(image_url, models)
    # Output the result as JSON
    print(json.dumps(result, default=convert_numpy_types))

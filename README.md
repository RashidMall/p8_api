# Image Mask Prediction App

This is a simple Flask application for performing image mask predictions using a pre-trained deep learning model.

## Overview

The app provides an HTTP API endpoint for making predictions. It accepts a base64-encoded image in JSON format and returns the predicted mask in JSON format.

## Requirements

- Python 3.7+  
- Flask  
- NumPy  
- TensorFlow  
- Pillow (PIL)  
- OpenCV  
- Base64  
- Requests  

You can find a full list of requirements in the `requirements.txt` file.  

## Usage

1. Clone the repository to your local machine:  

```bash  
git clone https://github.com/your-username/image-mask-prediction-app.git  
cd image-mask-prediction-app  
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask application locally:
```bash
python app.py
```

4. The app will start running on `http://127.0.0.1:5060/`. You can now send POST requests to the `/predict_mask` endpoint to get mask predictions.

## API Endpoint
* Endpoint: `/predict_mask`
* Method: POST
* Input: JSON object with a base64-encoded image
* Output: JSON object with the predicted mask

Example request:  
{
    "image": "base64_encoded_image_data_here"
}

Example response:  
{
    "mask": "base64_encoded_mask_data_here"
}

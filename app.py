from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 128

# Load the trained model
model = load_model('final_optimized_model.h5', compile=False)

# Define a route for mask prediction
@app.route('/predict_mask', methods=['POST'])
def predict_mask():
	try:
		# Get the base64-encoded image from the request
		data = request.get_json()
		if 'image' in data:
			image_data = data['image']
			image_data = base64.b64decode(image_data)

			# Convert the image data to an image
			image = Image.open(io.BytesIO(image_data))

			# Ensure it's an image file
			if image.format in ('JPEG', 'PNG', 'GIF', 'BMP'):
				# Resize the image to the specified dimensions
				image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

				# Convert the image to a numpy array
				image_array = np.array(image)

				# Make predictions using the model
				predicted_mask = model.predict(np.expand_dims(image_array, axis=0))

				# Post-process the predicted mask
				encoded_mask = base64.b64encode(predicted_mask)

				# Return the predicted mask as JSON or any other format
				return jsonify({'mask': encoded_mask.decode("utf-8")})

			else:
				return jsonify({'error': 'Invalid image format'})

		else:
			return jsonify({'error': 'No image data provided'})

	except Exception as e:
		return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5060)
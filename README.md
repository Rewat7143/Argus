# Argus 

This repository provides a simple yet effective method for detecting deepfake images using MesoNet, a convolutional neural network architecture. MesoNet is specifically designed for deepfake detection and has shown promising results in various studies.

## Installation

To use the deepfake detection model provided in this repository, you can install the required dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
```

## Usage

After installing the required dependencies, you can use the provided MesoNet model for deepfake detection. Here's a basic usage example:

```python
from deepfake_detection import Meso4

# Instantiate the MesoNet model
meso = Meso4()

# Load pre-trained weights
meso.load_weights('/path/to/pretrained/weights.h5')

# Prepare image data (assuming 'data' folder contains images)
data_generator = ImageDataGenerator(rescale=1./255)
generator = data_generator.flow_from_directory(
    'data',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')

# Make predictions
X, y = generator.next()
prediction = meso.predict(X)

# Display results
print("Predicted likelihood:", prediction[0][0])
print("Actual label:", int(y[0]))
```

## Model Architecture

The MesoNet architecture used in this repository consists of multiple convolutional layers followed by max-pooling, batch normalization, and dropout layers. The final layer uses a sigmoid activation function to output a probability score indicating the likelihood of an image being a deepfake.

## Contributing

Contributions to this repository are welcome! If you have ideas for improvements or encounter any issues, feel free to open an issue or submit a pull request.

## Credits

This repository is maintained by the following team members:

- Vinod Polinati - Machine Learning
- Reddy Rewat - Machine Learning
- Shaik Shajid - UI/UX Designer
- Miazur Rahaman - Streamlit Developer

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

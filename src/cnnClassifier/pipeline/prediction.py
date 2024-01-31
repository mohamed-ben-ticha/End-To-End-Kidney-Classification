import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
from cnnClassifier.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts/training", "model.h5"))

        config = ConfigurationManager()
        training_config = config.get_training_config()
        imagename = self.filename
        test_image = image.load_img(
            imagename, target_size=training_config.params_image_size[:-1]
        )
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = "Tumor"
            return [{"image": prediction}]
        else:
            prediction = "Normal"
            return [{"image": prediction}]

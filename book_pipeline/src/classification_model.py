from typing import List, Any

import numpy as np


class LayoutClassificationModel():
    def __init__(self, model_name: str, classifier: Any, scaler: Any, label_encoder: Any):
        """Initializes the layout classification model with the given parameters.
        Args:
            model_name (str): Name of the model.
            classifier (Any): The trained classifier object.
            scaler (Any): Scaler used for feature normalization.
            label_encoder (Any): Label encoder for converting labels to integers.
        """
        self.model_name = model_name
        self.classifier = classifier
        self.scaler = scaler
        self.label_encoder = label_encoder
    
    def predict(self, features: List[float]) -> str:
        """Predicts the layout category for the given features.
        Args:
            features (List[float]): List of feature values.
        Returns:
            str: Predicted layout category.
        """
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0.0, inplace=True)
        scaled_features = self.scaler.transform(features)
        prediction = self.classifier.predict(scaled_features)
        return self.label_encoder.inverse_transform(prediction)[0]
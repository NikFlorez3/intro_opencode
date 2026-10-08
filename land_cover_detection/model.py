import torch
import numpy as np
from pathlib import Path


class PrithviLandCoverModel:
    MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification"

    BAND_NAMES = ["Blue", "Green", "Red", "Narrow_NIR", "SWIR_1", "SWIR_2"]

    CLASS_NAMES = [
        "Natural_Vegetation",
        "Forest",
        "Corn",
        "Soybeans",
        "Wetlands",
        "Developed_Barren",
        "Open_Water",
        "Winter_Wheat",
        "Alfalfa",
        "Fallow_Idle_Cropland",
        "Cotton",
        "Sorghum",
        "Other",
    ]

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load(self):
        try:
            from torchgeo.models import PrithviBackbone
        except ImportError:
            raise ImportError(
                "torchgeo is required. Install with: pip install torchgeo"
            )

        self.model = PrithviBackbone.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, input_array):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if isinstance(input_array, np.ndarray):
            input_array = torch.from_numpy(input_array).float()

        input_array = input_array.to(self.device)

        if input_array.ndim == 3:
            input_array = input_array.unsqueeze(0)

        with torch.no_grad():
            features = self.model(input_array)

        return features

    def get_class_names(self):
        return self.CLASS_NAMES

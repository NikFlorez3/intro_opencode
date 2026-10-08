import numpy as np
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds


class LandCoverInference:
    CLASS_MAPPING = {
        "Forest": "Bosque",
        "Natural_Vegetation": "Vegetacion_natural",
        "Corn": "Agricultura",
        "Soybeans": "Agricultura",
        "Winter_Wheat": "Agricultura",
        "Alfalfa": "Agricultura",
        "Fallow_Idle_Cropland": "Agricultura",
        "Cotton": "Agricultura",
        "Sorghum": "Agricultura",
        "Open_Water": "Agua",
        "Developed_Barren": "Urbano_Suelo_desnudo",
        "Wetlands": "Humedales",
        "Other": "Otros",
    }

    GENERAL_CLASSES = [
        "Bosque",
        "Vegetacion_natural",
        "Agricultura",
        "Agua",
        "Urbano_Suelo_desnudo",
        "Humedales",
        "Otros",
    ]

    def __init__(self, model):
        self.model = model

    def run(self, input_array):
        predictions = self.model.predict(input_array)
        return predictions

    def map_to_general_classes(self, predictions, class_names):
        mapped = np.zeros_like(predictions)
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for orig_name, general_name in self.CLASS_MAPPING.items():
            if orig_name in class_to_idx and general_name in self.GENERAL_CLASSES:
                orig_idx = class_to_idx[orig_name]
                general_idx = self.GENERAL_CLASSES.index(general_name)
                mapped[predictions == orig_idx] = general_idx

        return mapped

    def compute_statistics(self, mapped_predictions):
        total_pixels = mapped_predictions.size
        stats = {}

        for idx, class_name in enumerate(self.GENERAL_CLASSES):
            count = np.sum(mapped_predictions == idx)
            percentage = (count / total_pixels) * 100
            stats[class_name] = {
                "pixels": int(count),
                "percentage": round(percentage, 2),
            }

        return stats

    def save_raster(self, mapped_predictions, profile, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mapped_predictions.astype(np.uint8), 1)

        return output_path

    def run_full_pipeline(self, input_array, profile, output_dir="output"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = self.run(input_array)
        class_names = self.model.get_class_names()
        mapped = self.map_to_general_classes(predictions, class_names)
        stats = self.compute_statistics(mapped)

        raster_path = self.save_raster(
            mapped, profile, output_dir / "land_cover_classification.tif"
        )

        return {
            "predictions": mapped,
            "statistics": stats,
            "raster_path": str(raster_path),
            "class_names": self.GENERAL_CLASSES,
        }

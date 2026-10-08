import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import requests
from datetime import datetime, timedelta
from pathlib import Path


class Sentinel2Preprocessor:
    COPERNICUS_API = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    S2_BAND_MAP = {
        "B02": "Blue",
        "B03": "Green",
        "B04": "Red",
        "B8A": "Narrow_NIR",
        "B11": "SWIR_1",
        "B12": "SWIR_2",
    }

    def __init__(self, bbox, start_date, end_date, cloud_cover=20):
        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_cover = cloud_cover

    def search_scenes(self):
        min_lon, min_lat, max_lon, max_lat = self.bbox
        footprint = (
            f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
            f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
        )

        query = (
            f"{self.COPERNICUS_API}?"
            f"$filter=Collection/Name eq 'SENTINEL-2' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}') and "
            f"ContentDate/Start ge {self.start_date}T00:00:00.000Z and "
            f"ContentDate/Start le {self.end_date}T23:59:59.999Z and "
            f"Online eq true and "
            f"CloudCover/percentage le {self.cloud_cover}"
            f"&$orderby=ContentDate/Start desc"
            f"&$top=10"
        )

        response = requests.get(query, timeout=30)
        response.raise_for_status()
        products = response.json().get("value", [])
        return products

    def download_band(self, product_id, band_name, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{product_id}_{band_name}.tif"

        if output_path.exists():
            return output_path

        url = (
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/"
            f"Products({product_id})/Nodes({band_name})/$value"
        )

        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path

    def create_temporal_composites(self, band_files, num_timesteps=3):
        if not band_files:
            raise ValueError("No band files provided")

        with rasterio.open(band_files[0]) as src:
            profile = src.profile.copy()
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs

        num_bands = len(self.S2_BAND_MAP)
        composite = np.zeros((num_timesteps, num_bands, height, width), dtype=np.float32)

        for t_idx, timestep_files in enumerate(band_files):
            for b_idx, (s2_band, _) in enumerate(self.S2_BAND_MAP.items()):
                matching_files = [f for f in timestep_files if s2_band in str(f)]
                if not matching_files:
                    continue

                with rasterio.open(matching_files[0]) as src:
                    data = src.read(1).astype(np.float32)
                    data = np.clip(data, 0, 10000) / 10000.0
                    composite[t_idx, b_idx] = data

        return composite, profile

    def normalize_for_model(self, composite, target_size=224):
        num_timesteps, num_bands, height, width = composite.shape

        pad_h = max(0, target_size - height)
        pad_w = max(0, target_size - width)
        composite = np.pad(
            composite,
            ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

        if height > target_size or width > target_size:
            composite = composite[:, :, :target_size, :target_size]

        mmseg_input = np.zeros((num_timesteps * num_bands, target_size, target_size), dtype=np.float32)
        for t in range(num_timesteps):
            for b in range(num_bands):
                mmseg_input[t * num_bands + b] = composite[t, b]

        return mmseg_input

    def preprocess(self, output_dir="data/sentinel2"):
        scenes = self.search_scenes()
        if not scenes:
            raise RuntimeError("No Sentinel-2 scenes found for the given criteria")

        band_files = []
        for scene in scenes[:3]:
            scene_bands = []
            for s2_band in self.S2_BAND_MAP.keys():
                path = self.download_band(scene["Id"], s2_band, output_dir)
                scene_bands.append(path)
            band_files.append(scene_bands)

        composite, profile = self.create_temporal_composites(band_files)
        mmseg_input = self.normalize_for_model(composite)

        return mmseg_input, profile, scenes

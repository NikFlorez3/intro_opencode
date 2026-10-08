import json
from pathlib import Path
from datetime import datetime


def save_statistics(stats, output_path, metadata=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
    }

    if metadata:
        output["metadata"] = metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output_path


def load_bbox_from_geojson(geojson_path):
    import json

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data["type"] == "FeatureCollection":
        coords = data["features"][0]["geometry"]["coordinates"][0]
    elif data["type"] == "Feature":
        coords = data["geometry"]["coordinates"][0]
    else:
        coords = data["coordinates"][0]

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    return [min(lons), min(lats), max(lons), max(lats)]


def validate_bbox(bbox):
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 values: [min_lon, min_lat, max_lon, max_lat]")

    min_lon, min_lat, max_lon, max_lat = bbox

    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90")
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Invalid bbox: min must be less than max")

    return bbox


def get_colombia_regions():
    return {
        "Amazonía": [-79.0, -4.5, -66.5, 1.5],
        "Andina": [-76.5, 0.5, -72.0, 7.5],
        "Caribe": [-77.5, 7.5, -71.0, 12.5],
        "Orinoquía": [-73.5, 3.5, -66.5, 8.5],
        "Pacífica": [-79.0, 1.0, -75.5, 7.5],
        "Antioquia": [-77.0, 5.5, -73.5, 9.0],
        "Cundinamarca": [-74.8, 3.8, -73.0, 5.8],
        "Valle del Cauca": [-77.5, 3.0, -75.5, 5.0],
    }

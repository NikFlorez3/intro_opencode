import argparse
import sys
from pathlib import Path

from land_cover_detection.model import PrithviLandCoverModel
from land_cover_detection.preprocessing import Sentinel2Preprocessor
from land_cover_detection.inference import LandCoverInference
from land_cover_detection.visualization import LandCoverVisualizer
from land_cover_detection.utils import (
    save_statistics,
    load_bbox_from_geojson,
    validate_bbox,
    get_colombia_regions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificación de cobertura terrestre en Colombia usando Prithvi (NASA/IBM)"
    )

    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box: min_lon min_lat max_lon max_lat",
    )
    parser.add_argument(
        "--geojson",
        type=str,
        help="Ruta a un archivo GeoJSON con el polígono de la zona de estudio",
    )
    parser.add_argument(
        "--region",
        type=str,
        help="Región predefinida de Colombia (ej: Amazonía, Andina, Caribe, Orinoquía, Pacífica, Antioquia, Cundinamarca, Valle del Cauca)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Fecha de inicio (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Fecha de fin (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cloud-cover",
        type=int,
        default=20,
        help="Porcentaje máximo de nubosidad (0-100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directorio de salida",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cuda, cpu). Auto-detección si no se especifica.",
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="Listar regiones predefinidas de Colombia",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_regions:
        regions = get_colombia_regions()
        print("\nRegiones predefinidas de Colombia:")
        print("-" * 50)
        for name, bbox in regions.items():
            print(f"  {name}: {bbox}")
        print()
        return 0

    if args.geojson:
        bbox = load_bbox_from_geojson(args.geojson)
    elif args.region:
        regions = get_colombia_regions()
        if args.region not in regions:
            print(f"Error: Región '{args.region}' no encontrada.")
            print(f"Regiones disponibles: {', '.join(regions.keys())}")
            return 1
        bbox = regions[args.region]
    elif args.bbox:
        bbox = args.bbox
    else:
        print("Error: Debe especificar --bbox, --geojson o --region")
        print("Use --list-regions para ver las regiones predefinidas")
        return 1

    try:
        bbox = validate_bbox(bbox)
    except ValueError as e:
        print(f"Error en bbox: {e}")
        return 1

    print(f"\nZona de estudio: {bbox}")
    print(f"Periodo: {args.start_date} a {args.end_date}")
    print(f"Nubosidad máxima: {args.cloud_cover}%")
    print(f"Directorio de salida: {args.output_dir}")
    print()

    print("Cargando modelo Prithvi...")
    model = PrithviLandCoverModel(device=args.device)
    model.load()
    print(f"Modelo cargado en: {model.device}")
    print()

    print("Buscando y descargando imágenes Sentinel-2...")
    preprocessor = Sentinel2Preprocessor(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_cover=args.cloud_cover,
    )

    try:
        input_tensor, profile, scenes = preprocessor.preprocess()
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        return 1

    print(f"Imágenes descargadas: {len(scenes)} escenas")
    print(f"Tensor de entrada: {input_tensor.shape}")
    print()

    print("Ejecutando inferencia...")
    inference = LandCoverInference(model)
    results = inference.run_full_pipeline(
        input_tensor=input_tensor,
        profile=profile,
        output_dir=args.output_dir,
    )

    print("\nResultados:")
    print("-" * 50)
    for class_name, stats in results["statistics"].items():
        print(f"  {class_name}: {stats['percentage']:.2f}% ({stats['pixels']} píxeles)")
    print()

    visualizer = LandCoverVisualizer()

    map_path = visualizer.plot_classification(
        predictions=results["predictions"],
        class_names=results["class_names"],
        output_path=Path(args.output_dir) / "land_cover_map.png",
        title=f"Cobertura terrestre - {args.start_date} a {args.end_date}",
    )
    print(f"Mapa guardado: {map_path}")

    stats_path = visualizer.plot_statistics(
        statistics=results["statistics"],
        output_path=Path(args.output_dir) / "land_cover_statistics.png",
        title=f"Distribución de coberturas - {args.start_date} a {args.end_date}",
    )
    print(f"Estadísticas guardadas: {stats_path}")

    json_path = save_statistics(
        stats=results["statistics"],
        output_path=Path(args.output_dir) / "land_cover_statistics.json",
        metadata={
            "bbox": bbox,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "cloud_cover": args.cloud_cover,
            "model": model.MODEL_ID,
        },
    )
    print(f"JSON guardado: {json_path}")

    print(f"\nRaster clasificado: {results['raster_path']}")
    print("\n¡Proceso completado exitosamente!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

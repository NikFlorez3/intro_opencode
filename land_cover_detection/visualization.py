import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class LandCoverVisualizer:
    CLASS_COLORS = {
        "Bosque": "#228B22",
        "Vegetacion_natural": "#90EE90",
        "Agricultura": "#FFD700",
        "Agua": "#1E90FF",
        "Urbano_Suelo_desnudo": "#808080",
        "Humedales": "#00CED1",
        "Otros": "#8B4513",
    }

    def __init__(self):
        self.cmap = None
        self.norm = None

    def _create_colormap(self, class_names):
        from matplotlib.colors import ListedColormap, BoundaryNorm

        colors = [self.CLASS_COLORS.get(c, "#000000") for c in class_names]
        self.cmap = ListedColormap(colors)
        self.norm = BoundaryNorm(
            np.arange(-0.5, len(class_names) + 0.5, 1), self.cmap.N
        )

    def plot_classification(self, predictions, class_names, output_path, title=None):
        self._create_colormap(class_names)

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        im = ax.imshow(predictions, cmap=self.cmap, norm=self.norm, interpolation="nearest")

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xlabel("Columna")
        ax.set_ylabel("Fila")

        patches = [
            mpatches.Patch(
                color=self.CLASS_COLORS.get(c, "#000000"),
                label=c.replace("_", " "),
            )
            for c in class_names
        ]
        ax.legend(
            handles=patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=True,
            title="Cobertura",
        )

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def plot_statistics(self, statistics, output_path, title=None):
        classes = list(statistics.keys())
        percentages = [statistics[c]["percentage"] for c in classes]
        colors = [self.CLASS_COLORS.get(c, "#000000") for c in classes]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.barh(classes, percentages, color=colors, edgecolor="black", linewidth=0.5)

        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("Porcentaje (%)")
        ax.set_ylabel("Cobertura terrestre")
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, max(percentages) * 1.15)

        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    def create_interactive_map(self, predictions, class_names, output_path):
        import folium
        from folium import plugins

        h, w = predictions.shape
        center_lat, center_lon = 4.5709, -74.2973

        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        colors = [self.CLASS_COLORS.get(c, "#000000") for c in class_names]

        img_overlay = folium.raster_layers.ImageOverlay(
            image=self._predictions_to_rgba(predictions, colors),
            bounds=[[center_lat - h * 0.0001, center_lon - w * 0.0001],
                    [center_lat + h * 0.0001, center_lon + w * 0.0001]],
            opacity=0.7,
        )
        img_overlay.add_to(m)

        legend_html = "<div style='position:fixed;bottom:50px;left:50px;z-index:1000;background-color:white;padding:10px;border:2px solid grey;border-radius:5px;'>"
        legend_html += "<b>Cobertura terrestre</b><br>"
        for c in class_names:
            color = self.CLASS_COLORS.get(c, "#000000")
            legend_html += f"<i style='background:{color};width:12px;height:12px;display:inline-block;margin-right:5px;'></i>{c.replace('_', ' ')}<br>"
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))

        return output_path

    def _predictions_to_rgba(self, predictions, colors):
        import matplotlib.colors as mcolors

        h, w = predictions.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        for idx, color in enumerate(colors):
            rgb = mcolors.to_rgb(color)
            mask = predictions == idx
            rgba[mask, 0] = int(rgb[0] * 255)
            rgba[mask, 1] = int(rgb[1] * 255)
            rgba[mask, 2] = int(rgb[2] * 255)
            rgba[mask, 3] = 180

        return rgba

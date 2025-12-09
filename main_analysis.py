import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg') # Fix for "main thread is not in main loop" error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import median_filter
import os

# --- Configuration ---
DATA_DIR = "data_v2"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dynamic World Classes
CLASSES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded Veg",
    4: "Crops",
    5: "Shrub/Scrub",
    6: "Built Area",
    7: "Bare Ground",
    8: "Snow/Ice"
}

# Colors for visualization
COLORS = np.array([
    [0.25, 0.61, 0.87], # Water (Blue)
    [0.22, 0.49, 0.28], # Trees (Dark Green)
    [0.53, 0.69, 0.33], # Grass (Light Green)
    [0.48, 0.53, 0.78], # Flooded Veg (Purple)
    [0.89, 0.59, 0.21], # Crops (Orange)
    [0.87, 0.76, 0.35], # Shrub (Yellow)
    [0.77, 0.16, 0.11], # Built Area (Red)
    [0.65, 0.61, 0.56], # Bare Ground (Gray)
    [1.00, 1.00, 1.00]  # Snow (White)
])

def load_image(path):
    with rasterio.open(path) as src:
        data = src.read()
        # (Bands, H, W) -> (H, W, Bands)
        data = np.moveaxis(data, 0, -1)
        return data, src.profile

def calculate_indices(img):
    """
    Calculate spectral indices to improve classification accuracy.
    Input: (H, W, 6) -> [Blue, Green, Red, NIR, SWIR1, SWIR2]
    """
    # Avoid division by zero
    epsilon = 1e-8
    
    blue  = img[:, :, 0]
    green = img[:, :, 1]
    red   = img[:, :, 2]
    nir   = img[:, :, 3]
    swir1 = img[:, :, 4]
    
    # NDVI (Vegetation)
    ndvi = (nir - red) / (nir + red + epsilon)
    
    # NDBI (Built-up)
    ndbi = (swir1 - nir) / (swir1 + nir + epsilon)
    
    # MNDWI (Water)
    mndwi = (green - swir1) / (green + swir1 + epsilon)
    
    # Stack features: Original 6 bands + 3 indices
    features = np.dstack([img, ndvi, ndbi, mndwi])
    return features

def save_plot(data, title, filename, rgb_img=None):
    plt.figure(figsize=(10, 10))
    
    if rgb_img is not None:
        # Display RGB Composite (Red, Green, Blue) -> Indices 2, 1, 0
        display_data = np.dstack([rgb_img[:,:,2], rgb_img[:,:,1], rgb_img[:,:,0]])
        
        # Normalize 0-1 for display
        p2, p98 = np.percentile(display_data, (2, 98))
        display_data = np.clip((display_data - p2) / (p98 - p2), 0, 1)
        
        plt.imshow(display_data)
    else:
        # Classification Map
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(COLORS)
        plt.imshow(data, cmap=cmap, vmin=0, vmax=8, interpolation='nearest')
        
        # Add legend
        patches = [plt.Rectangle((0,0),1,1, color=COLORS[i]) for i in range(9)]
        plt.legend(patches, CLASSES.values(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

def analyze_data():
    print("Starting Analysis Pipeline (Using Ground Truth Data)...")
    
    urban_stats = []
    years = range(2018, 2026)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for year in years:
        lc_path = os.path.join(DATA_DIR, f"LandCover_{year}.tif")
        s2_path = os.path.join(DATA_DIR, f"Sentinel2_{year}.tif")
        
        if not os.path.exists(lc_path):
            print(f"Skipping Year {year}: Data not found.")
            continue
            
        print(f"   Processing Year {year}...")
        
        # Load Land Cover (Ground Truth)
        # Dynamic World labels are already classified!
        lc_data, profile = load_image(lc_path)
        lc_data = lc_data.squeeze() # (H, W)
        
        # Calculate Urban Area (Class 6)
        urban_pixels = np.sum(lc_data == 6)
        urban_pct = (urban_pixels / lc_data.size) * 100
        urban_stats.append((year, urban_pct))
        print(f"     -> Urban Coverage: {urban_pct:.2f}%")
        
        # Save Visualization (Map)
        save_plot(lc_data, f"LULC Classification {year}", f"map_{year}.png")
        
        # Save RGB Reference (if available)
        if os.path.exists(s2_path):
            raw_img, _ = load_image(s2_path)
            save_plot(None, f"Satellite Image {year}", f"satellite_{year}.png", rgb_img=raw_img)
        
        # Save Classification TIF (Copy of Ground Truth)
        profile.update(count=1, dtype=rasterio.uint8)
        with rasterio.open(os.path.join(OUTPUT_DIR, f"classification_{year}.tif"), 'w', **profile) as dst:
            dst.write(lc_data.astype(rasterio.uint8), 1)

    # 4. Generate Change Map
    print("\nGenerating Change Map...")
    if len(urban_stats) >= 2:
        start_year = urban_stats[0][0]
        end_year = urban_stats[-1][0]
        
        with rasterio.open(os.path.join(OUTPUT_DIR, f"classification_{start_year}.tif")) as src:
            lc_start = src.read(1)
        with rasterio.open(os.path.join(OUTPUT_DIR, f"classification_{end_year}.tif")) as src:
            lc_end = src.read(1)
            
        change_map = np.zeros_like(lc_start)
        # Highlight new urban areas
        change_map[(lc_start != 6) & (lc_end == 6)] = 1
        
        plt.figure(figsize=(10, 10))
        plt.imshow(change_map, cmap='Reds', interpolation='nearest')
        plt.title(f"Urban Expansion ({start_year}-{end_year})")
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, "change_map.png"), bbox_inches='tight')
        print("Change map saved.")

    # 5. Generate Report
    print("\nGenerating Report...")
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("URBAN LULC CHANGE DETECTION REPORT\n")
        f.write("==================================\n\n")
        f.write("Methodology: Analysis of Google Dynamic World V1 Dataset\n")
        f.write("Classes: Dynamic World Schema (9 classes)\n\n")
        f.write("Yearly Urban Area Statistics:\n")
        f.write("-----------------------------\n")
        for year, pct in urban_stats:
            f.write(f"{year}: {pct:.2f}% Urban Coverage\n")
        
        if len(urban_stats) >= 2:
            growth = urban_stats[-1][1] - urban_stats[0][1]
            f.write(f"\nTotal Urban Growth ({urban_stats[0][0]}-{urban_stats[-1][0]}): {growth:+.2f}%\n")
            
    print(f"Report saved to {report_path}")
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    analyze_data()

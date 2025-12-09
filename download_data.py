import ee
import requests
import os
import zipfile
import io
PROJECT_ID = 'formidable-code-280411' 

def init_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=PROJECT_ID)
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print("Google Earth Engine not authenticated. Triggering authentication...")
        ee.Authenticate()
        ee.Initialize()
        print("Google Earth Engine initialized successfully.")

def download_image(image, scale, region, filename):
    """Download an image from GEE using getDownloadURL."""
    try:
        url = image.getDownloadURL({
            'scale': scale,
            'region': region,
            'format': 'GEO_TIFF'
        })
        print(f"Downloading from: {url[:50]}...")
        
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Saved to {filename}")
            return True
        else:
            print(f"Failed to download. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    init_gee()
    
    output_dir = "data_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define Region of Interest (ROI) - Bangalore Center
    point = ee.Geometry.Point([77.64, 13.05])
    roi = point.buffer(7500).bounds()
    region = roi.getInfo()['coordinates']
    
    years = range(2018, 2026)
    
    print(f"\nStarting download for years: {list(years)}")
    print(f"Region: Bangalore (5km buffer)")
    print("-" * 50)
    
    for year in years:
        print(f"\nProcessing Year: {year}")
        
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .median() \
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']) \
            .clip(roi)
            
        s2_filename = os.path.join(output_dir, f"Sentinel2_{year}.tif")
        print("Fetching Sentinel-2 image...")
        download_image(s2, 10, region, s2_filename)
        
        dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .select('label') \
            .mode() \
            .clip(roi)
            
        dw_filename = os.path.join(output_dir, f"LandCover_{year}.tif")
        print("Fetching Dynamic World label...")
        download_image(dw, 10, region, dw_filename)

    print("\n" + "="*50)
    print("All downloads complete!")
    print(f"Data saved in: {os.path.abspath(output_dir)}")
    print("="*50)

if __name__ == "__main__":
    main()

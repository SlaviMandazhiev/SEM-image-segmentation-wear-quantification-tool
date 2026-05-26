import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans

from typing import List, Optional, Tuple


flaeche_pixel = 90000/946729

fixed_boundary = 5

offset_value = 7

adjust_fe = 20

adjust_w = 15

folder_path = os.path.dirname(os.path.abspath(__file__)) #erstellt den  Pfad zum Ordner, in dem das Skript liegt




#Das ist die eingentliche Funktion, die die Pixel zählt
#min_area is the minimum region size to keep (in pixels)
#fill_largest_component determines if holes in the final mask will be filled (keep in mind that this may be buggy as it uses flood_fill which may sometimes overfill the mask if the largest component is not well defined)
def count_pixels(ubergabe_image, lower_color: int, upper_color: int, min_area: int, fill_largest_component: bool):

    # Create a mask based on the specified color range
    mask = cv2.inRange(ubergabe_image, lower_color, upper_color)
    
    # Set kernel
    kernel = np.ones((2, 2), np.uint8)

    # 1. Fill small holes (close)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. Remove small speckles (open)
    mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=4)
    
    #if only a background is detected, it gracefully stops the function.
    if num_labels <= 1:
        return np.zeros_like(mask_cleaned)
    

    filtered_mask = np.zeros_like(mask_cleaned)

    # Loop through components and keep only big ones
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255
    
    if fill_largest_component == True:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background
        largest_component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
        #fill holes in the largest component
        flood_mask = np.zeros((largest_component.shape[0] + 2, largest_component.shape[1] + 2), np.uint8)
        flood_filled = largest_component.copy()
        cv2.floodFill(flood_filled, flood_mask, seedPoint=(0, 0), newVal=255)
        holes = cv2.bitwise_not(flood_filled)
        filled_largest_component = cv2.bitwise_or(largest_component, holes)
        
        filtered_mask[filled_largest_component == 255] = 255
    
   
    return filtered_mask
    

def get_intensity_boundaries_from_histogram(hist: np.ndarray, adjust_w: int, adjust_fe: int, offset_value: int, k: int = 4, random_state: int = 42) -> Optional[List[int]]:
    """
    Compute intensity boundaries between k classes from a 1D histogram using k-means clustering.

    Parameters:
        hist (np.ndarray): 1D array of shape (256,) representing pixel counts for each intensity.
        k (int): Number of clusters.
        random_state (int): Seed for reproducibility (default: 42).
    """
    try:
        #check if input is a valid 1D histogram
        if not isinstance(hist, np.ndarray):
            raise TypeError("Input histogram must be a NumPy array.")
        if hist.shape == (256, 1):
            hist = hist.ravel()
        elif hist.shape != (256,):
            raise ValueError("Input histogram must be of shape (256,) or (256, 1).")
            
        if k < 2 or k > 256:
            raise ValueError("k must be between 2 and 256.")

        #prepare intensity values and sample weights
        intensity_values = np.arange(1, 256).reshape(-1, 1).astype(np.float32)
        sample_weights = hist[1:].astype(np.float32)

        #run kmeans
        kmeans = KMeans(n_clusters = k-1, random_state=random_state, n_init=10)
        kmeans.fit(intensity_values, sample_weight=sample_weights)

        #get sorted cluster centers
        centers = np.sort(kmeans.cluster_centers_.ravel())

        #compute midpoints as boundaries (this is because k-means separates points based on Euclidean distances)
        dynamic_boundaries = [int(round((centers[i] + centers[i + 1]) / 2)) for i in range(len(centers) - 1)]
        
        boundaries = [int(offset_value)] + dynamic_boundaries
        
        if k >= 4:  # 3-material mode: second-to-last = Ti/Fe boundary, last = Fe/W boundary
            boundaries[-1] = min(boundaries[-1] + adjust_w, 255)
            boundaries[-2] = min(boundaries[-2] + adjust_fe, 255)
        else:       # 2-material mode: only one dynamic boundary = Ti/Fe
            boundaries[-1] = min(boundaries[-1] + adjust_fe, 255)

        
        print(boundaries)

        return boundaries, centers

    except Exception as e:
        print(f"Error in get_intensity_boundaries_from_histogram: {e}")
        return None
    


    
def create_scale_plots_histogram(gray_image, scale_power: float, plot: bool = False):
    all_pix = gray_image.shape[0]*gray_image.shape[1]
    hist1 = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    #histogramm normalization
    hist_normalized = hist1 / hist1.sum()
    
    hist_adjusted = np.power(hist_normalized.ravel(), scale_power)
    
    # Histogramm plotten
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(hist_adjusted, color='black')
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Intensity (0–255)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return all_pix, hist_adjusted


def build_tool_mask(
    image_gray: np.ndarray,
    fixed_boundary: int = 40,
    offset_value: int = 7,
    kernel_size: Tuple[int, int] = (2, 2),
    debug_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a solid tool mask from a grayscale SEM image and create an adjusted image
    where tool pixels are offset brighter and background is forced to 0.

    Returns:
        adjusted_image: uint8 grayscale, background 0, tool pixels offset by offset_value
        solid_tool_mask: uint8 mask (0/255) of tool including filled holes
        tool_mask: uint8 mask (0/255) of tool region before hole filling (largest CC)
    """

    if image_gray is None or image_gray.ndim != 2:
        raise ValueError("build_tool_mask expects a 2D grayscale image (uint8).")

    # 1) Rough foreground mask (non-background)
    _, rough_mask = cv2.threshold(image_gray, fixed_boundary, 255, cv2.THRESH_BINARY)

    # 2) Morphological closing to connect regions
    kernel = np.ones(kernel_size, np.uint8)
    closed = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel)

    # 3) Keep largest connected component (assumed tool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)

    if num_labels <= 1:
        # No foreground detected -> return empty masks and zeroed adjusted image
        tool_mask = np.zeros_like(image_gray, dtype=np.uint8)
        solid_tool_mask = tool_mask.copy()
        adjusted_image = np.zeros_like(image_gray, dtype=np.uint8)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "Tool_Mask.jpg"), tool_mask)
            cv2.imwrite(os.path.join(debug_dir, "solid_tool_mask.jpg"), solid_tool_mask)
            cv2.imwrite(os.path.join(debug_dir, "adjusted_image.jpg"), adjusted_image)

        return adjusted_image, solid_tool_mask, tool_mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    tool_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # 4) Force top and left borders black for flood fill stability
    tool_mask[:1, :] = 0
    tool_mask[:, :1] = 0

    # 5) Fill holes inside tool region (flood fill from outside)
    flood_mask = np.zeros((tool_mask.shape[0] + 2, tool_mask.shape[1] + 2), np.uint8)
    flood_filled = tool_mask.copy()
    cv2.floodFill(flood_filled, flood_mask, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood_filled)
    solid_tool_mask = cv2.bitwise_or(tool_mask, holes)

    # 6) Build adjusted image:
    #    - start from original gray
    #    - brighten tool pixels by offset_value
    #    - clamp to [0, 255]
    #    - force non-tool pixels to 0
    adjusted_image = image_gray.copy()
    tool_pixels = (solid_tool_mask == 255)

    adjusted_image[tool_pixels] = np.clip(
        adjusted_image[tool_pixels].astype(np.int16) + int(offset_value),
        0, 255
    ).astype(np.uint8)

    # Force background to 0 explicitly
    adjusted_image[~tool_pixels] = 0

    # Optional: if you still want to zero out anything "too dark" inside tool area
    # (keep this only if it truly helps your downstream segmentation)
    adjusted_image[adjusted_image < offset_value] = 0

    # 7) Optional debug saving (kept OUT of the core logic unless requested)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "Tool_Mask.jpg"), tool_mask)
        cv2.imwrite(os.path.join(debug_dir, "solid_tool_mask.jpg"), solid_tool_mask)
        cv2.imwrite(os.path.join(debug_dir, "adjusted_image.jpg"), adjusted_image)

    return adjusted_image, solid_tool_mask, tool_mask

def fill_dark_speckles(
    img: np.ndarray,
    tool_mask: np.ndarray,
    se_size: int = 9,     # structuring element (odd): 7–15 typical
    thresh_mode: str = "otsu",     # "otsu" or "percentile"
    perc: float = 95.0,            # percentile if thresh_mode == "percentile"
    max_area: int = 1000,          # remove only small pits
    inpaint_radius: int = 3,       # 2–4 px works well
    debug_dir: Optional[str] = None
) -> np.ndarray:
    """
    Remove dark speckles inside bright regions using morphological black-hat.
    Operates only inside tool_mask. Returns a despeckled copy of img.
    """
    assert img.ndim == 2 and tool_mask.ndim == 2

    # 1) Black-hat highlights dark specks (closing - image)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    blackhat = cv2.subtract(closed, img)

    # Only consider inside the tool
    bh_inside = cv2.bitwise_and(blackhat, blackhat, mask=tool_mask)

    # 2) Threshold black-hat to get candidate speckles
    if thresh_mode.lower() == "otsu":
        # Otsu on nonzero bh values inside the tool
        vals = bh_inside[tool_mask == 255]
        t = 0
        if vals.size > 0:
            # Otsu needs an image; build a tiny hist-based threshold
            _t, th = cv2.threshold(bh_inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            th = np.zeros_like(bh_inside)
    else:
        # Percentile threshold (robust when Otsu under/over-segments)
        vals = bh_inside[tool_mask == 255]
        if vals.size == 0:
            th = np.zeros_like(bh_inside)
        else:
            t = np.percentile(vals, perc)
            _, th = cv2.threshold(bh_inside, int(t), 255, cv2.THRESH_BINARY)

    # 3) Keep only small components = true speckles
    num, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    specks = np.zeros_like(th)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            specks[lab == i] = 255

    # 4) Optional light open to clean stray pixels
    k = np.ones((3,3), np.uint8)
    specks = cv2.morphologyEx(specks, cv2.MORPH_OPEN, k, iterations=1)

    # 5) Fill speckles (choose one)
    # a) Edge-preserving: inpaint from neighborhood
    out = cv2.inpaint(img, specks, inpaint_radius, cv2.INPAINT_TELEA)

    # b) Or replace with local mean (comment the line above and uncomment below):
    # local_mean = cv2.blur(img, (7,7), borderType=cv2.BORDER_REFLECT)
    # out = img.copy(); out[specks == 255] = local_mean[specks == 255]

    # 6) Keep background at 0
    out[tool_mask == 0] = 0

    # ---- Debug saves (optional) ----
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, "dbg_blackhat.png"), blackhat)
        cv2.imwrite(os.path.join(debug_dir, "dbg_bh_inside.png"), bh_inside)
        cv2.imwrite(os.path.join(debug_dir, "dbg_specks.png"), specks)

    return out

def image_detector(folder_path: str) -> list:
    file_names = []
    for file in os.listdir(folder_path): #liest alle Dateinamen im Ordner aus und speichert sie in der Liste
        if file.endswith(".jpeg") or file.endswith(".jpg"):
            file_names.append(file)
    print(file_names)
    
    return file_names

def folder_creator(folder_path: str, name: str) -> str:
    folder_dir = os.path.join(folder_path, os.path.splitext(name)[0])
    os.makedirs(folder_dir, exist_ok=True)

    src = os.path.join(folder_path, name)
    dst = os.path.join(folder_dir, name)
    shutil.copy(src, dst)
    time.sleep(0.1)
    
    return folder_dir

def process_one_image(image_path: str,
                      adjust_fe: int,
                      adjust_w: int,
                      offset_value: int,
                      fixed_boundary: int,
                      n_materials: int = 3,
                      scale_power: float = 0.1,
                      save_dir: str = None,
                      debug_dir: Optional[str] = None,
                      plot_hist: bool = False,) -> dict:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    adjusted_image_speckles, solid_tool_mask, tool_mask = build_tool_mask(image_gray, fixed_boundary=fixed_boundary, offset_value=offset_value, debug_dir=debug_dir)
    
    adjusted_image = fill_dark_speckles(adjusted_image_speckles, solid_tool_mask)
        
    all_pix, hist_adjusted = create_scale_plots_histogram(adjusted_image, scale_power, plot=plot_hist) #SCALE HISTOGRAM HERE
        
    k = n_materials + 1
    boundaries, centers = get_intensity_boundaries_from_histogram(hist_adjusted, offset_value=offset_value, adjust_fe=adjust_fe, adjust_w=adjust_w, k=k) #PERFORM K-MEANS HERE

    retval, img_thresh = cv2.threshold(adjusted_image, boundaries[0], 255, cv2.THRESH_BINARY)
    number_of_white_pix = np.sum(img_thresh == 255)

    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, "Erkannte_Flaeche.jpg"), img_thresh)

    return {
        "adjusted_image": adjusted_image,
        "tool_mask": solid_tool_mask,
        "boundaries": boundaries,
        "centers": centers,
        "histogram": hist_adjusted,
        "threshold_image": img_thresh,
        "white_pixels": number_of_white_pix,
        "all_pixels": all_pix
        }

def mask_generator(adjusted_image: np.ndarray, boundaries: list, save_dir: str = None) -> Tuple[dict, np.ndarray]:

    three_material_mode = len(boundaries) >= 3

    # ── Coating mask (Ti, darkest, always present) ────────────────────────
    mask_titan = count_pixels(adjusted_image, boundaries[0], boundaries[1], 250, False)
    ti_pixel_count = cv2.countNonZero(mask_titan)

    # ── Adhesion wear mask (Fe) ────────────────────────────────────────────
    if three_material_mode:
        mask_eisen = count_pixels(adjusted_image, boundaries[1], boundaries[2], 50, False)
    else:
        # 2-material mode: Adhesion covers everything above the Coating boundary
        mask_eisen = count_pixels(adjusted_image, boundaries[1], 255, 50, False)
    fe_pixel_count = cv2.countNonZero(mask_eisen)

    # ── Substrate mask (W, brightest, 3-material mode only) ───────────────
    wo_pixel_count = 0
    if three_material_mode:
        mask_wolfram = count_pixels(adjusted_image, boundaries[2], 255, 1000, True)
        wo_pixel_count = cv2.countNonZero(mask_wolfram)
        # Substrate takes priority: remove its pixels from Adhesion mask
        mask_eisen[mask_wolfram == 255] = 0
        fe_pixel_count = cv2.countNonZero(mask_eisen)

    # ── Color overlay ──────────────────────────────────────────────────────
    color_image = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)
    color_image[mask_titan == 255] = [255, 0, 0]   # Blue  = Coating
    color_image[mask_eisen == 255] = [0, 0, 255]   # Red   = Adhesion
    if three_material_mode:
        color_image[mask_wolfram == 255] = [0, 255, 0]  # Green = Substrate

    # ── Optional save ──────────────────────────────────────────────────────
    if save_dir:
        print(f"\nCoating pixels in folder {save_dir}: {ti_pixel_count}")
        time.sleep(0.5)
        cv2.imwrite(os.path.join(save_dir, "Coating.jpg"), mask_titan)
        time.sleep(0.5)

        print(f"\nAdhesion pixels in folder {save_dir}: {fe_pixel_count}")
        time.sleep(0.5)
        cv2.imwrite(os.path.join(save_dir, "Adhesion.jpg"), mask_eisen)
        time.sleep(0.5)

        if three_material_mode:
            print(f"\nSubstrate pixels in folder {save_dir}: {wo_pixel_count}")
            time.sleep(0.5)
            cv2.imwrite(os.path.join(save_dir, "Substrate.jpg"), mask_wolfram)
            time.sleep(0.5)

        cv2.imwrite(os.path.join(save_dir, "Combined_Mask.jpg"), color_image)
        time.sleep(0.5)

    return {
        "wo_pixel_count": wo_pixel_count,
        "fe_pixel_count": fe_pixel_count,
        "ti_pixel_count": ti_pixel_count
        }, color_image
    
def auswertung_file_creator(auswertung_file_path: str, image_process_results: dict, pixel_counts: dict, pixel_area: float = flaeche_pixel):
    
    all_pix = image_process_results["all_pixels"]
    number_of_white_pix = image_process_results["white_pixels"]
    centers = image_process_results["centers"]
    boundaries = image_process_results["boundaries"]
    
    wo_pixel_count = pixel_counts["wo_pixel_count"]
    fe_pixel_count = pixel_counts["fe_pixel_count"]
    ti_pixel_count = pixel_counts["ti_pixel_count"]
    
    
    three_material_mode = len(boundaries) >= 3

    with open(auswertung_file_path, 'w') as auswertung_file:
         auswertung_file.write(f"*************************************************************\n")
         auswertung_file.write(f"Anzahl der gesamten Pixel: {all_pix}\n")
         auswertung_file.write(f"Anzahl der Pixel des Objekts: {number_of_white_pix}\n")
         auswertung_file.write(f"Das Objekt bedekt {number_of_white_pix*100/all_pix:.3f}% des Gesamtbilds\n")
         auswertung_file.write(f"Liste der 'Cluster Centers': {centers}\n")

         auswertung_file.write(f"------------------------------------------------------------\n")
         auswertung_file.write(f"Coating pixels: {ti_pixel_count}\n")
         Titan_Anteil=(ti_pixel_count*100/number_of_white_pix)
         auswertung_file.write(f"Coating area (% of tool): {Titan_Anteil:.3f} %\n")
         auswertung_file.write(f"Intensity range (Coating): {boundaries[0]} - {boundaries[1]}\n")
         auswertung_file.write(f"------------------------------------------------------------\n")

         if three_material_mode:
             fe_upper = boundaries[2]
         else:
             fe_upper = 255

         auswertung_file.write(f"------------------------------------------------------------\n")
         auswertung_file.write(f"Adhesion wear pixels: {fe_pixel_count}\n")
         Eisen_Anteil=(fe_pixel_count*100/number_of_white_pix)
         auswertung_file.write(f"Adhesion wear area (% of tool): {Eisen_Anteil:.3f} %\n")
         auswertung_file.write(f"Intensity range (Adhesion): {boundaries[1]} - {fe_upper}\n")
         auswertung_file.write(f"------------------------------------------------------------\n")

         if three_material_mode:
             auswertung_file.write(f"------------------------------------------------------------\n")
             auswertung_file.write(f"Substrate pixels: {wo_pixel_count}\n")
             Wolfram_Anteil=(wo_pixel_count*100/number_of_white_pix)
             auswertung_file.write(f"Substrate area (% of tool): {Wolfram_Anteil:.3f} %\n")
             auswertung_file.write(f"Intensity range (Substrate): {boundaries[2]} - 255\n")
             auswertung_file.write(f"------------------------------------------------------------\n")
         else:
             Wolfram_Anteil = 0.0

         auswertung_file.write(f"------------------------------------------------------------\n")
         auswertung_file.write(f"Absolute area Coating  = {(ti_pixel_count*pixel_area):.4f} µm²\n")
         auswertung_file.write(f"Absolute area Adhesion = {(fe_pixel_count*pixel_area):.4f} µm²\n")
         if three_material_mode:
             auswertung_file.write(f"Absolute area Substrate = {(wo_pixel_count*pixel_area):.4f} µm²\n")
         auswertung_file.write(f"------------------------------------------------------------\n")

         auswertung_file.write(f"\nUndetected area: {100-Titan_Anteil-Wolfram_Anteil-Eisen_Anteil:.2f} %\n")

         auswertung_file.write(f"*************************************************************\n")







def main_pipeline(
    folder_path: str = folder_path,
    adjust_fe: int = adjust_fe,
    adjust_w: int = adjust_w,
    offset_value: int = offset_value,
    fixed_boundary: int = fixed_boundary,
) -> None:
    """
    Batch-process all JPEG images found in folder_path.

    For each image:
      1. Create a dedicated output sub-folder.
      2. Run the full segmentation pipeline (tool isolation, despeckling,
         adaptive thresholding).
      3. Generate and save individual material masks and combined overlay.
      4. Write a text report (Auswertung.txt).
    """
    file_names = image_detector(folder_path)
    if not file_names:
        print("No JPEG images found in: " + folder_path)
        return

    for name in file_names:
        print("")
        print("=" * 50)
        print("Processing: " + name)

        image_path = os.path.join(folder_path, name)
        save_dir   = folder_creator(folder_path, name)

        results = process_one_image(
            image_path     = image_path,
            adjust_fe      = adjust_fe,
            adjust_w       = adjust_w,
            offset_value   = offset_value,
            fixed_boundary = fixed_boundary,
            save_dir       = save_dir,
        )

        pixel_counts, _ = mask_generator(
            results["adjusted_image"],
            results["boundaries"],
            save_dir = save_dir,
        )

        report_path = os.path.join(save_dir, "Auswertung.txt")
        auswertung_file_creator(report_path, results, pixel_counts)

        print("  -> Saved to: " + save_dir)

    print("")
    print("Done. Processed " + str(len(file_names)) + " image(s).")


if __name__ == "__main__":
    main_pipeline()

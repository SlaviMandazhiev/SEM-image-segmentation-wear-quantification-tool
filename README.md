# SEM Image Segmentation, Wear Quantification Tool

A desktop GUI tool for segmenting SEM (Scanning Electron Microscope) images of coated cutting tool surfaces and quantifying the area of each wear region.

The tool identifies and measures three material regions:
- **Coating**: the PVD coating layer (darkest in SEM)
- **Adhesion wear**: adhered workpiece material (intermediate intensity)
- **Substrate**: exposed substrate material (brightest in SEM)

For a full explanation of the algorithm, segmentation approach, and results, see the project report (PDF).

---

## Installation

```bash
pip install -r requirements.txt
```

`tkinter` is part of the Python standard library. If it is missing on Linux, install it via your package manager (e.g. `sudo apt install python3-tk`).

---

## Usage

Both files must be in the same folder. Launch the GUI with:

```bash
python RSA_Seg_GUI.py
```

**Workflow:**
1. **Open Image**: load a JPEG SEM image
2. **Set Scale**: draw a line over the scale bar and enter its real length in µm to calibrate pixel-to-area conversion
3. **Set Crop Region**: optionally enter pixel ranges to crop out the scale bar or irrelevant areas
4. **Run Analysis**: segments the image automatically using adaptive K-means clustering on the grayscale histogram
5. **Adjust Boundaries**: drag the intensity sliders and click Apply to fine-tune the segmentation
6. **Save Results**: writes the wear masks, colour overlay, and a text report with area percentages to a chosen folder

# 🗂️ YOLO-Gemini Flashcard Pipeline

An automated, end-to-end data pipeline using **YOLOv8** and the **Gemini Vision API** to extract, semantically filter, and format bilingual flashcards into perfectly mirrored A4 PDFs. 

The final output is a print-ready, double-sided grid (images on the front, vocabulary text on the back) designed to be printed and manually cut into physical, tangible flashcards.

## 🚀 The Problem
Creating physical flashcards from a 40-page German language textbook manually requires hours of cropping, translating, and aligning images in a document. 

This project completely automates the workflow:

1. **Detects** the flashcard regions on raw textbook pages.

2. **Reads** the German text and dynamically splits the image from the word.

3. **Packs** the extracted fronts and backs into a mirrored A4 grid so they align perfectly when printed double-sided.

## 🛠️ Architecture & Tech Stack

* **Computer Vision (YOLOv8):** A custom-trained YOLO model scans the raw textbook pages and crops out individual flashcard candidates.
* **Vision-Language Model (Gemini 2.5 Flash-Lite):** Acts as both an OCR and an intelligent slicer. It reads the German text and returns a JSON object containing the exact vertical `split_ratio` to separate the illustration from the vocabulary word.
* **Image Processing (Python/Pillow):** Resizes, pads, and stitches the extracted images onto a high-resolution white A4 canvas.

---

## 💡 Technical Challenges 

Building a fully automated physical printing pipeline introduced several unique data and API constraints. 

### 1. The "Semantic Filter" (Handling Noisy Data & Edge Cases)
**The Problem:** Because the dataset consisted of raw photos of a textbook, YOLO occasionally cropped unintended elements—such as page margins, random textbook page numbers, a thumb holding the page down, or even completely blank areas. 
**The Solution:** We engineered the Gemini prompt to act as a **Semantic Filter** based purely on OCR presence. If Gemini analyzes a YOLO crop and does not find a valid printed German noun or phrase at the bottom, it returns `{"split_ratio": null, "text": null}`. The script catches this, flags the image as noise, and moves it to a `rejected_samples/` folder, ensuring the final PDF is 100% clean while preserving valid number cards.

### 2. Overcoming API Rate Limits (State Persistence)
**The Problem:** The Gemini Free Tier API has strict usage quotas, which caused the script to crash (`429 Quota Exceeded`) midway through the 400+ card dataset.
**The Solution:** We implemented a robust persistence and throttling mechanism. The script uses a file-existence check to skip previously processed images, allowing the batch to be safely "resumed" across multiple days without duplicating API calls. A `time.sleep()` throttle ensures it respects the requests-per-minute ceiling.

### 3. The Mirrored Grid Algorithm (Physical Printing)
**The Problem:** If you place a front image in the top-left corner of Page 1, the printer flips the paper horizontally. If the back text is also in the top-left of Page 2, they will not align.
**The Solution:** We wrote a routing algorithm that mathematically mirrors the grid for the back pages. By calculating `back_col = (COLS - 1) - col`, the script automatically places the text in the top-right slot to perfectly back the top-left image.

---

## 📂 Repository Structure

```text
yolo-gemini-flashcard-pipeline/
│
├── data/
│   ├── flashcards.yaml          # YOLO dataset configuration
│   └── sample_pages/            # Raw textbook page inputs
│
├── models/
│   └── yolo_flashcard.pt        # Custom trained YOLOv8 weights
│
├── scripts/
│   ├── 01_yolo_cli.py           # Full CLI for YOLO training, splitting, and extraction
│   ├── 02_process_gemini.py     # Gemini API split and Semantic Filter logic
│   └── 03_generate_pdf.py       # A4 double-sided grid generation
│
├── output/
│   ├── sample_output/           # Visual examples of the pipeline (Raw -> Front/Back)
│   ├── rejected_samples/        # Examples of noise automatically filtered by Gemini
│   └── Printable_Flashcards.pdf # Final, print-ready output
│
├── requirements.txt             # Python dependencies
└── .gitignore                   # Hidden files and API keys
```

## ⚙️ How to Run

### 1. Installation & Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

Add your Google API key to a `.env` file at the root of the project:
```text
GEMINI_API_KEY=your_api_key_here
```

### 2. The YOLO Pipeline (Step 1)
The `01_yolo_cli.py` script is a complete command-line interface managing the YOLO lifecycle. 

*(Optional) If training from scratch:*
```bash
python scripts/01_yolo_cli.py split --images data/images --labels data/labels --output data/dataset
python scripts/01_yolo_cli.py train --data data/flashcards.yaml --epochs 30
```

**To extract flashcards from new pages:**
```bash
python scripts/01_yolo_cli.py batch --model models/yolo_flashcard.pt --input data/sample_pages --output output_cards
```

### 3. The Gemini Semantic Filter (Step 2)
Process the YOLO crops, dynamically split them, and automatically filter out non-vocabulary noise.
```bash
python scripts/02_process_gemini.py
```

### 4. Generate the Physical PDF (Step 3)
Stitch the validated fronts and backs into a mirrored A4 grid for physical printing.
```bash
python scripts/03_generate_pdf.py
```

---
*Built and presented for the Build & Learn: Data Science with Coffee Meetup (Berlin).*

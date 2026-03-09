import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# 1. Load the API key from the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Could not find GEMINI_API_KEY in your .env file!")

# Configure Gemini
genai.configure(api_key=api_key)

# We use gemini-2.0-flash as it is the fastest and best for this task
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def process_single_card(card_path: Path, front_dir: Path, back_dir: Path):
    """Sends a single cropped card to Gemini, splits it, and saves the pieces."""
    
    img = Image.open(card_path)
    
    # The prompt explicitly asks for JSON
    prompt = """Analyze this German flashcard image.
    A valid flashcard MUST have printed German vocabulary text at the bottom (e.g., "vierzig", "der Gürtel", "backen"). 
    1. If you see printed German text at the bottom, it is VALID. Return the JSON with the split_ratio and the text:
       {"split_ratio": 0.65, "text": "the actual german text"}
    2. If there is NO German text at the bottom (e.g., it is just a book fold, a stray mark, or a picture with no words), it is INVALID. Return exactly:
       {"split_ratio": null, "text": null}    
    Format: {"split_ratio": 0.65, "text": "der Arzt"}"""
    
    #1. Find the German text at the bottom.
    #2. Determine the exact vertical ratio (between 0.0 and 1.0) where the picture ends and the text begins. 
    #Return ONLY a valid JSON object in this exact format, with no markdown formatting:{"split_ratio": 0.65, "text": "der Arzt"}"""

    print(f"Asking Gemini to analyze: {card_path.name}...")
    
    try:
        # Request strict JSON output to prevent code crashes
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        result = json.loads(response.text)
        split_ratio = result.get("split_ratio", 0.7) # default to 0.7 if it fails
        german_text = result.get("text", "Unknown Text")
        
        # Split at detected ratio
        w, h = img.size
        split_y = int(h * split_ratio)
        
        front = img.crop((0, 0, w, split_y))      # Top = Image only
        back = img.crop((0, split_y, w, h))       # Bottom = Text only
        
        # Save the separated files
        base_name = card_path.stem
        front.save(front_dir / f"{base_name}_front.png")
        back.save(back_dir / f"{base_name}_back.png")
        
        # We can also save the text to a file so you can use it later!
        with open(back_dir / f"{base_name}_text.txt", "w", encoding="utf-8") as f:
            f.write(german_text)
            
        print(f"  ✅ Success! Extracted text: '{german_text}'")
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ Failed to process {card_path.name}: {e}")
        # --- NEW: Safely stop if we hit the daily quota ---
        if "429" in error_msg or "Quota exceeded" in error_msg:
            return "QUOTA_HIT"
        return False
def main():
    # Set your folders here
    yolo_output_dir = Path("output_cards") # Where YOLO saved the crops
    final_fronts_dir = Path("final_flashcards/fronts")
    final_backs_dir = Path("final_flashcards/backs")
    
    # Create the final folders if they don't exist
    final_fronts_dir.mkdir(parents=True, exist_ok=True)
    final_backs_dir.mkdir(parents=True, exist_ok=True)

    # Find all PNGs in the YOLO output folder
    # rglob allows it to search inside subfolders if YOLO created them
    all_images = list(yolo_output_dir.rglob("*.png"))
    
    if not all_images:
        print(f"No images found in {yolo_output_dir}. Did you run the YOLO script first?")
        return

    for img_path in all_images:
        # CRITICAL: Skip the visualization files!
        if img_path.name.endswith("_detected.png"):
            continue
        base_name = img_path.stem
        expected_front = final_fronts_dir / f"{base_name}_front.png"
        
        # --- NEW: THE "RESUME" CHECK ---
        # If the front image is already in the final folder, we skip it completely!
        if expected_front.exists():
            print(f"⏭️ Skipping {img_path.name} (Already processed)")
            continue   
        # Process the individual card
        status=process_single_card(img_path, final_fronts_dir, final_backs_dir)
        
        if status == "QUOTA_HIT":
            print("\n🛑 Daily quota reached! The script has safely stopped.")
            print("Run it again tomorrow to pick up exactly where you left off.")
            sys.exit(0)
        # Wait 5 seconds between requests to stay within free tier (15 RPM) limit
        time.sleep(5)

if __name__ == "__main__":
    main()
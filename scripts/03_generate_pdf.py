import math
from pathlib import Path
from PIL import Image, ImageDraw

def create_flashcard_pdf():
    fronts_dir = Path("final_flashcards/fronts")
    backs_dir = Path("final_flashcards/backs")
    output_pdf = Path("final_flashcards/Printable_Flashcards_A4.pdf")
    
    # A4 Paper Dimensions at 300 DPI (High Quality Print)
    A4_W, A4_H = 2480, 3508
    
    # Grid Settings (3 columns, 3rows = 9 cards per page)
    COLS, ROWS = 3, 3
    CARDS_PER_PAGE = COLS * ROWS
    
    # Symmetrical margins are REQUIRED for double-sided printing to align
    MARGIN_X, MARGIN_Y = 140, 140 
    
    SLOT_W = (A4_W - 2 * MARGIN_X) // COLS
    SLOT_H = (A4_H - 2 * MARGIN_Y) // ROWS

    front_files = sorted(list(fronts_dir.glob("*_front.png")))
    if not front_files:
        print("No front images found! Check your folders.")
        return

    print(f"Found {len(front_files)} cards. Generating A4 PDF...")

    pdf_pages = []

    # Process cards in chunks of 12 (one A4 sheet front+back)
    for i in range(0, len(front_files), CARDS_PER_PAGE):
        chunk = front_files[i : i + CARDS_PER_PAGE]
        
        # Create blank white A4 canvases for front and back
        front_page = Image.new('RGB', (A4_W, A4_H), 'white')
        back_page = Image.new('RGB', (A4_W, A4_H), 'white')
        
        draw_front = ImageDraw.Draw(front_page)
        draw_back = ImageDraw.Draw(back_page)

        for index, front_path in enumerate(chunk):
            base_name = front_path.stem.replace("_front", "")
            back_path = backs_dir / f"{base_name}_back.png"
            
            if not back_path.exists():
                print(f"Warning: Missing back for {base_name}. Skipping this card.")
                continue

            # Calculate Row and Column (0-indexed)
            row = index // COLS
            col = index % COLS
            
            # THE MAGIC TRICK: Reverse the column for the back page so it mirrors perfectly
            back_col = (COLS - 1) - col

            # Calculate exact pixel coordinates on the A4 page
            front_x = MARGIN_X + (col * SLOT_W)
            front_y = MARGIN_Y + (row * SLOT_H)
            
            back_x = MARGIN_X + (back_col * SLOT_W)
            back_y = MARGIN_Y + (row * SLOT_H)

            # Open, resize, and paste the front image into its slot
            f_img = Image.open(front_path)
            f_img.thumbnail((SLOT_W - 20, SLOT_H - 20)) # Leave a small 10px padding
            f_offset_x = front_x + (SLOT_W - f_img.width) // 2
            f_offset_y = front_y + (SLOT_H - f_img.height) // 2
            front_page.paste(f_img, (f_offset_x, f_offset_y))

            # Open, resize, and paste the back image into its mirrored slot
            b_img = Image.open(back_path)
            b_img.thumbnail((SLOT_W - 20, SLOT_H - 20))
            b_offset_x = back_x + (SLOT_W - b_img.width) // 2
            b_offset_y = back_y + (SLOT_H - b_img.height) // 2
            back_page.paste(b_img, (b_offset_x, b_offset_y))

            # Draw light gray cut-lines around the slots
            draw_front.rectangle([front_x, front_y, front_x + SLOT_W, front_y + SLOT_H], outline="lightgray", width=2)
            draw_back.rectangle([back_x, back_y, back_x + SLOT_W, back_y + SLOT_H], outline="lightgray", width=2)

        pdf_pages.append(front_page)
        pdf_pages.append(back_page)

    # Save all generated pages into a single multi-page PDF
    if pdf_pages:
        pdf_pages[0].save(
            output_pdf, 
            save_all=True, 
            append_images=pdf_pages[1:], 
            resolution=300.0
        )
        print(f"\n✅ Success! Your PDF is ready at: {output_pdf}")
        print(f"Generated {len(pdf_pages)} pages (Fronts and Backs).")

if __name__ == "__main__":
    create_flashcard_pdf()
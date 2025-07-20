from fpdf import FPDF
import os

def generate_output(layout_data, output_path):
    """
    Generates a PDF file with recognized text placed according to detected regions.
    Args:
        layout_data (dict): Contains 'text_regions' with 'bbox' and 'text'.
        output_path (str): Path to save the PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    # pdf.set_font("Arial", size=12)

    # Add a Unicode font (NotoSans-Regular.ttf must be in your project directory or specify full path)
    font_path = os.path.join(os.path.dirname(__file__), "NotoSans-Regular.ttf")
    pdf.add_font("NotoSans", "", font_path, uni=True)
    pdf.set_font("NotoSans", size=12)

    for region in layout_data["text_regions"]:
        x, y, w, h = region["bbox"]
        text = region.get("text", "")
        # Place text roughly at the detected block position
        pdf.set_xy(x / 5, y / 5)  # Scale coordinates for PDF page
        pdf.multi_cell(w / 5, 10, text)

    pdf.output(output_path)
import markdown
import os
import sys

walkthrough_path = r"C:\Users\adity\.gemini\antigravity\brain\c8b68576-2982-4a08-b1cc-0ed5448eebb8\walkthrough.md"
export_path = r"c:\Users\adity\.gemini\antigravity\scratch\nifty-dashboard-py\analysis_2026\Nifty_Quant_Analytics_Report.html"

try:
    with open(walkthrough_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Basic CSS for a beautiful clean report
    css = """
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 40px; background-color: #f9f9f9; }
        .container { background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #2c3e50; border-bottom: 1px solid #ebebeb; padding-bottom: 10px; margin-top: 30px; }
        h1 { color: #1a252f; font-size: 2.2em; text-align: center; border-bottom: 2px solid #3498db; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.95em; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        blockquote { background-color: #eef2f5; border-left: 5px solid #3498db; margin: 20px 0; padding: 15px; font-style: italic; }
        code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 4px; font-family: monospace; }
        .print-btn { display: block; margin: 20px auto; padding: 10px 20px; background-color: #27ae60; color: white; text-align: center; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 200px; }
        @media print { .print-btn { display: none; } body { background-color: white; padding: 0; } .container { box-shadow: none; padding: 0; } }
    </style>
    """

    # We use basic replace since we might not have `markdown` library installed on user's machine
    # But let's try `markdown` library first
    html_content = ""
    try:
        html_content = markdown.markdown(md_text, extensions=['tables'])
    except Exception as e:
        # Fallback if markdown library is missing
        html_content = f"<pre style='white-space: pre-wrap;'>{md_text}</pre>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Nifty Quant Analytics Report</title>
        {css}
    </head>
    <body>
        <div class="container">
            <button class="print-btn" onclick="window.print()">Export to PDF</button>
            {html_content}
        </div>
    </body>
    </html>
    """

    with open(export_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Successfully generated HTML report at: {export_path}")
    print("User can open this file and click 'Export to PDF' (Ctrl+P -> Save as PDF).")
    
except Exception as e:
    print(f"Failed to generate report: {e}")

from utils.report_generator import generate_pdf_from_md
dirty_markdown = '''# Mock Report 🚨
**Recommendation**: STRONG BUY 🟢
The company scores an exceptional 95/100, driven by ₹15k Cr revenue and 25% margins.
Management says “we expect 2x growth” — a stellar guidance… 🚀'''

try:
    pdf_bytes = generate_pdf_from_md(dirty_markdown)
    print(f'SUCCESS: PDF generated {len(pdf_bytes)} bytes.')
except Exception as e:
    print(f'CRASH: {str(e)}')

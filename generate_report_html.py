"""
Generates a standalone, beautifully styled HTML document from House_Rent_Prediction_Report.md
with inline styling suitable for printing or saving to PDF via Chrome/Edge.
"""

import os
import re

md_path = 'House_Rent_Prediction_Report.md'
html_path = 'House_Rent_Prediction_Report.html'

if not os.path.exists(md_path):
    raise FileNotFoundError(f"{md_path} not found")

with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert markdown basic elements to clean HTML
# 1. Convert headers
text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)

# 2. Bold and Italics
text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

# 3. Code blocks
text = re.sub(r'```(?:bash|python)?\n(.*?)```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

# 4. Images
text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<div class="img-container"><img src="\2" alt="\1"><div class="caption">\1</div></div>', text)

# 5. Tables
def parse_markdown_table(match):
    lines = match.group(0).strip().split('\n')
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|')[1:-1]]
    
    html = ['<table class="styled-table">', '<thead>', '<tr>']
    for h in headers:
        html.append(f'<th>{h}</th>')
    html.extend(['</tr>', '</thead>', '<tbody>'])
    
    for row_line in lines[2:]:
        if not row_line.strip():
            continue
        cells = [c.strip() for c in row_line.split('|')[1:-1]]
        html.append('<tr>')
        for c in cells:
            html.append(f'<td>{c}</td>')
        html.append('</tr>')
    html.extend(['</tbody>', '</table>'])
    return '\n'.join(html)

table_regex = re.compile(r'(\|.+?\|\n\|[-: |]+?\|\n(?:\|.+?\|\n?)+)', re.MULTILINE)
text = table_regex.sub(parse_markdown_table, text)

# 6. Paragraphs and bullet points
paragraphs = text.split('\n\n')
formatted_paragraphs = []
for p in paragraphs:
    p = p.strip()
    if not p:
        continue
    if p.startswith('<h') or p.startswith('<table') or p.startswith('<pre') or p.startswith('<div class="img-container"'):
        formatted_paragraphs.append(p)
    elif p.startswith('- ') or p.startswith('* '):
        items = p.split('\n')
        list_html = ['<ul>']
        for item in items:
            item_clean = re.sub(r'^[-*]\s+', '', item.strip())
            list_html.append(f'<li>{item_clean}</li>')
        list_html.append('</ul>')
        formatted_paragraphs.append('\n'.join(list_html))
    elif re.match(r'^\d+\.\s+', p):
        items = p.split('\n')
        list_html = ['<ol>']
        for item in items:
            item_clean = re.sub(r'^\d+\.\s+', '', item.strip())
            list_html.append(f'<li>{item_clean}</li>')
        list_html.append('</ol>')
        formatted_paragraphs.append('\n'.join(list_html))
    elif p == '---':
        formatted_paragraphs.append('<hr>')
    else:
        formatted_paragraphs.append(f'<p>{p}</p>')

body_html = '\n\n'.join(formatted_paragraphs)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>House Rent Prediction - Machine Learning Assignment Report</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.65;
        color: #1f2937;
        background-color: #f9fafb;
        margin: 0;
        padding: 40px 20px;
    }}
    .document-container {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 50px 60px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }}
    h1 {{
        color: #1e3a8a;
        font-size: 26px;
        font-weight: 800;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 12px;
        margin-top: 0;
    }}
    h2 {{
        color: #1e40af;
        font-size: 20px;
        font-weight: 700;
        margin-top: 36px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 8px;
    }}
    h3 {{
        color: #374151;
        font-size: 16px;
        font-weight: 600;
        margin-top: 24px;
    }}
    p, li {{
        font-size: 14.5px;
        color: #374151;
    }}
    ul, ol {{
        padding-left: 24px;
        margin-bottom: 16px;
    }}
    li {{
        margin-bottom: 6px;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #e5e7eb;
        margin: 30px 0;
    }}
    .styled-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13.5px;
    }}
    .styled-table th, .styled-table td {{
        border: 1px solid #e5e7eb;
        padding: 10px 12px;
        text-align: left;
    }}
    .styled-table th {{
        background-color: #f3f4f6;
        color: #111827;
        font-weight: 600;
    }}
    .styled-table tr:nth-child(even) {{
        background-color: #f9fafb;
    }}
    code {{
        background-color: #f3f4f6;
        color: #ef4444;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 13px;
        font-family: Consolas, monospace;
    }}
    pre {{
        background-color: #1f2937;
        color: #f9fafb;
        padding: 16px;
        border-radius: 6px;
        overflow-x: auto;
    }}
    pre code {{
        background: none;
        color: inherit;
        padding: 0;
    }}
    .img-container {{
        text-align: center;
        margin: 28px 0;
    }}
    .img-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }}
    .caption {{
        font-size: 12.5px;
        color: #6b7280;
        margin-top: 8px;
        font-style: italic;
    }}
    @media print {{
        body {{
            background: #ffffff;
            padding: 0;
        }}
        .document-container {{
            box-shadow: none;
            padding: 0;
            max-width: 100%;
        }}
        .img-container {{
            page-break-inside: avoid;
        }}
    }}
</style>
</head>
<body>
<div class="document-container">
{body_html}
</div>
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"Generated standalone publication HTML report: '{html_path}'")

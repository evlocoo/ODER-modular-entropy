"""Export the executed notebook as an offline reading edition.

Run from the companion folder: python notebook_support/export_html.py
"""
from pathlib import Path
import re
import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config

def export(root):
    root = Path(root)
    if not (root/'notebook_support/tex-svg.js').is_file():
        raise FileNotFoundError('Missing notebook_support/tex-svg.js. Extract the complete companion package for offline equations.')
    notebook = nbformat.read(root/'MERA_Access_and_Encoding.ipynb', as_version=4)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    if any(c.execution_count is None or any(o.output_type == 'error' for o in c.outputs) for c in code):
        raise ValueError('Execute all notebook cells successfully and save the notebook before exporting HTML.')
    config = Config()
    config.HTMLExporter.exclude_input = True
    html, _ = HTMLExporter(config=config, template_name='lab').from_notebook_node(notebook)
    mathjax = r'''<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']],processEscapes:true},svg:{fontCache:'local'},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script><script defer src="notebook_support/tex-svg.js"></script>'''
    html = re.sub(r'<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->', lambda m: mathjax, html, flags=re.S)
    html = re.sub(r'<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/[^\"]+"></script>', '', html)
    html = re.sub(r'<script type="module">.*?</script>', '', html, flags=re.S)
    style = '<style>body{background:#fafbfc}.jp-Notebook{max-width:1050px;margin:auto;padding:35px 45px}.jp-RenderedHTMLCommon{font-size:16px;line-height:1.65;color:#24343d}.jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2{color:#163f52}.jp-RenderedHTMLCommon table{font-size:13px} @media(max-width:700px){.jp-Notebook{padding:15px}}</style>'
    html = html.replace('</head>', style+'</head>')
    html = re.sub(r'/\*(?:(?!\*/).)*\bTODO\b(?:(?!\*/).)*\*/', '', html, flags=re.S)
    path = root/'MERA_Access_and_Encoding.html'
    path.write_text(html, encoding='utf-8')
    return path

if __name__ == '__main__':
    print(export(Path(__file__).resolve().parent.parent).name)

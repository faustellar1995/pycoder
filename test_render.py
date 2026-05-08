"""测试 markdown_renderer.py：将 .md 文件转换为 HTML 并输出"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from markdown_renderer import markdown_to_html

with open("test_markdown_input.md", "r", encoding="utf-8") as f:
    md_text = f.read()

html_output = markdown_to_html(md_text)
print(html_output)

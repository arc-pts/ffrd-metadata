from bs4 import BeautifulSoup
from IPython.display import HTML
import json
import re
import requests


def get_ld_json(url: str) -> dict:
    parser = "html.parser"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, parser)
    return json.loads("".join(soup.find("script", {"type": "application/ld+json"}).contents))


def stylized_json(data: dict) -> HTML:
    # json_data = json.dumps(data)
    # data = json.loads(json_data)

    # Define a regular expression to match URLs
    url_regex = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')

    orange = "#CC5500"
    violet = "#5C00A3"
    green = "#108f10"
    blue = "#0000EE"
    black = "#131413"

    # Define the CSS style
    css_style = """
        .json-link {
            color: #0000EE;
            text-decoration: none;
            border-bottom: 1px dashed #00aeff;
        }
    """

    # Define the HTML string with styling
    html_with_style = f"""
        <style>
            {css_style}
        </style>
        <pre>
    """

    # Define the indentation level for each nested level
    indentation = "    "

    # Recursively build the HTML string for the JSON data
    def build_html(data, level):
        html = ""
        if isinstance(data, dict):
            html += "{\n"
            for key, value in data.items():
                key_html = f'<span style="color: {green}">{indentation * level}"{key}"</span>: '
                value_html = build_html(value, level + 1)
                html += key_html + value_html + ",\n"
            html += f"{indentation * (level - 1)}}}"
        elif isinstance(data, list):
            html += "[\n"
            for value in data:
                value_html = build_html(value, level + 1)
                html += f"{indentation * level}{value_html},\n"
            html += f"{indentation * (level - 1)}]"
        elif isinstance(data, str):
            html += url_regex.sub(
                lambda match: f'<a class="json-link" href="{match.group(0)}">{match.group(0)}</a>',
                f'<span style="color: {black}">"{data}"</span>',
            )
        else:
            html += f'<span style="color: {orange}">{json.dumps(data)}</span>'
        return html

    # Build the HTML string for the JSON data
    html_with_style += build_html(data, 1)

    html_with_style += "</pre>"

    # Display the HTML string
    return HTML(html_with_style)

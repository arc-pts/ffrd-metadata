"""
This helper script executes a jupyter notebook, manages parameters, 
and prunes input cells/code from the exported html document
to keep focus on the content and not the inline code.
"""

import os
import papermill as pm
from nbconvert import get_exporter
from nbconvert.preprocessors import TagRemovePreprocessor


def run(nb_name: str = "ffrd-metadata-demo-2", params: dict = {}, mill_dir: str = "mill", remove_exec_nb: bool = False):
    """
    params, if passed, will collected stored params from the executed notebook
    This is useful if using papermill to run many notebooks in production
    and analytics from notebook outputs is desired
    """

    # Create a tmp dir for processing (in an attempt to not make a mess....)
    processing_dir = f"{mill_dir}"

    src_notebook = f"{nb_name}.ipynb"
    dst_notebook = f"{processing_dir}/{src_notebook}"
    dst_html = f"{processing_dir}/{nb_name}.html"

    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)

    # Run the source notebook
    pm.execute_notebook(src_notebook, dst_notebook, parameters=params)

    # Disable the ExtractOutput preprocessor. This prevents images embedded in
    # the notebook (ie: plots) from being externally linked
    config = {"ExtractOutputPreprocessor": {"enabled": False}}

    # Use HTML to export or optionally use the (archived) html_toc
    # to show table of contents
    HTMLExporter = get_exporter("html")
    exporter = HTMLExporter(config)

    # Add a preprocessor to the exporter to remove the notebook cells
    # with the tag 'remove_cell' from the output html document
    cell_remover = TagRemovePreprocessor(remove_cell_tags={"remove_cell"}, remove_input_tags={"remove_input"})

    exporter.register_preprocessor(cell_remover, True)

    # Generate HTML and write it to a file
    html, _ = exporter.from_filename(dst_notebook)
    with open(dst_html, "w") as f:
        f.write(html.encode('utf-8').decode('ascii', 'ignore'))

    # Clean up as desired
    if remove_exec_nb:
        os.remove(dst_notebook)

    print(f" --------- Sucess --------- ")


if __name__ == "__main__":
    run(remove_exec_nb=True)

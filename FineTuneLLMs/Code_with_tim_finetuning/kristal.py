import nbformat

nb = nbformat.read("./3_code_with_tim_simple_finetuning_llm.ipynb", as_version=4)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

nbformat.write(nb, "fixed_notebook.ipynb")

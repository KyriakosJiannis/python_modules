from typing import Callable
import requests
import importlib.util as util


def import_from_web(url: str, function_name: str) -> Callable:
    """
    Imports a function from a Python file hosted on web.

    :param url: The URL of the Python file.
    :param function_name: The name of the function to import from the file.
    :return: The imported function.

    Example usage:

    # Import the function 'my_function' from a GitHub file

    web_url = 'https://raw.githubusercontent.com/KyriakosJiannis/python_modules/main/python_modules/eda_charts.py'

    my_function = 'descriptive_dataframe'

    imported_function = import_from_github(web_url, my_function)

    descriptive_dataframe = imported_function()

    """

    # Fetch the content of the file
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download the code")

    # Define a new module
    module_name = 'remote_module'
    spec = util.spec_from_loader(module_name, loader=None)
    remote_module = util.module_from_spec(spec)

    # Execute the fetched Python code in the context of this new module
    exec(response.text, remote_module.__dict__)

    # Return the specific function
    return getattr(remote_module, function_name)


if __name__ == "__main__":
    pass

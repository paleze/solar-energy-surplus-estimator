import os
import logging
from pathlib import Path


# Get logger
app_logger = logging.getLogger("estimator-logger")
app_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

app_logger.addHandler(handler)




def get_secret_from_env_or_file(env_var=None, file_path_var=None):
    """
    Retrieve a secret value from an environment variable or a file.

    Args:
    - env_var (str): The name of the environment variable.
    - file_path_var (str): The name of the environment variable that contains the file path.

    Returns:
    - str: The secret value.
    """
    if env_var is not None:
        secret = os.getenv(env_var)
        if secret is not None:
            return secret

    if file_path_var is not None and os.path.exists(file_path_var):
        try:
            with open(file_path_var, 'r') as file:
                return file.read().strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read secret from file {file_path_var}: {e}") from e
    
    raise RuntimeError(f"Secret not found: neither {env_var} nor '{file_path_var}' is set.")




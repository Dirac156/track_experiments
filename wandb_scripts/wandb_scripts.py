import os
import random
import string

import wandb
from cryptography.fernet import Fernet

from typing import Dict, Any, Optional
from pathlib import Path

def generate_random_character() -> str:
  """"

  """
  return random.choices(string.ascii_lowercase)[0]

def generate_random_string(length: int = 10) -> str:
  """"
  """
  str = [generate_random_character() for _ in range(length)]
  return "".join(str)

# configure wandb folder
wandb_folder_path = Path("./config/")
wandb_folder_path.mkdir(parents=True, exist_ok=True)

fernet_encryption_key_path = wandb_folder_path / "fernet_encryption_key.txt"

wandb_api_path = wandb_folder_path / "wandb_api.txt"


def ensure_directory_exists(path: str):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_file(path: str, content: str = None) -> bytes:
    """Generate a new Fernet key and save it to a file.

    Args:
        path (str): The file path to save the Fernet key.

    Returns:
        bytes: The generated Fernet key.
    """
    with open(path, 'wb') as _file:
        _file.write(content)
    return path

def load_content(path: str) -> bytes:
    """Load the Fernet key from a file.

    Args:
        path (str): The file path where the Fernet key is stored.

    Returns:
        bytes: The loaded Fernet key.
    """
    with open(path, 'rb') as _file:
        content = _file.read()
    return content



def encrypt_api_key(api_key: str, secret_key: bytes) -> bytes:
    """Encrypt an API key using a Fernet key.

    Args:
        api_key (str): The API key to encrypt.
        secret_key (bytes): The Fernet key used for encryption.

    Returns:
        bytes: The encrypted API key.
    """
    fernet = Fernet(secret_key)
    return fernet.encrypt(api_key.encode())

def decrypt_api_key(encrypted_api_key: bytes, secret_key: bytes) -> str:
    """Decrypt an API key using a Fernet key.

    Args:
        encrypted_api_key (bytes): The encrypted API key.
        secret_key (bytes): The Fernet key used for decryption.

    Returns:
        str: The decrypted API key.
    """
    fernet = Fernet(secret_key)
    return fernet.decrypt(encrypted_api_key).decode()

def get_and_save_wandb_api_key(
    api_key: str,
    wandb_api_path: str = wandb_api_path,
    fernet_encryption_key_path: str = fernet_encryption_key_path
  ) -> None:
    """Encrypt and save the W&B API key.

    Args:
        api_key (str): The W&B API key to encrypt and save.
        wandb_api_path (str): Path to the file where the encrypted API key will be saved.
        fernet_encryption_key_path (str): Path to the file where the Fernet key will be saved.
    """
    # Generate a Fernet encryption key
    fernet_encryption_key = Fernet.generate_key()

    # Encrypt the API key
    encrypted_api_key = encrypt_api_key(api_key, fernet_encryption_key)

    # Save the encrypted API key and the Fernet encryption key to their respective files
    save_to_file(wandb_api_path, encrypted_api_key)
    save_to_file(fernet_encryption_key_path, fernet_encryption_key)

def load_and_decrypt_wandb_api_key(
    wandb_api_path: str = wandb_api_path,
    fernet_encryption_key_path: str = fernet_encryption_key_path
  ) -> str:
    """Load and decrypt the W&B API key.

    Args:
        wandb_api_path (str): Path to the file where the encrypted API key is stored.
        fernet_encryption_key_path (str): Path to the file where the Fernet key is stored.

    Returns:
        str: The decrypted W&B API key.
    """
    # Load the encrypted API key and the Fernet encryption key
    encrypted_api_key = load_content(wandb_api_path)
    fernet_encryption_key = load_content(fernet_encryption_key_path)

    # Decrypt and return the API key
    return decrypt_api_key(encrypted_api_key, fernet_encryption_key)

def get_wandb_api_key() -> str:
    """
    Retrieve the W&B API key.

    This function will first check if the W&B API key is available in the
    environment variables. If not, it will look for the API key in a specified file.
    If the key is still not found, it will prompt the user to input the key, save
    it in the file, and set it in the environment variables.

    Returns:
        str: The W&B API key.
    """
    WANDB_API_KEY = "WANDB_API_KEY"

    # Check if the API key is already in the environment variables
    api_key = os.getenv(WANDB_API_KEY)
    if api_key:
        return api_key

    # Check if the API key file exists and read the key
    if os.path.exists(wandb_api_path):
      api_key = load_and_decrypt_wandb_api_key()
      if api_key:
        # Set the API key in the environment variables
        os.environ['WANDB_API_KEY'] = api_key
        return api_key

    # Ask the user for the API key and save it in the file and environment variable
    api_key = input("Enter your W&B API key: ").strip()
    get_and_save_wandb_api_key(api_key)
    os.environ['WANDB_API_KEY'] = api_key
    return api_key


def init_wandb(
    project_name: str = "",
    tags: list[str] = ["baseline"],
    save_code: bool = True,
    reinit: bool = False,
    id: str = "",
) -> wandb.run:
    """Initialize a connection with Weights & Biases (W&B).

    This function initializes a W&B instance for a project, and the instance is responsible for syncing data between W&B and the training instance.

    Args:
        - project_name (str): A string representing the name of the project.
        - tags (list[str]): A list of tags associated with the project.
        - save_code (bool): Default is True. If True, save the script or notebook initializing the instance.
        - reinit (bool): If True, reinitialize the W&B instance.
        - id (str): A string representing a unique identifier for the run.

    Returns:
        wandb.run: The function returns the W&B run instance.

    Example:
        ```python
        run_instance = init_wandb(
            project_name="text_classification",
            tags=["paper2"],
            save_code=True,
            reinit=False,
            id="001"
        )
        ```

    """

    # generate a random project name if not provided
    if project_name == "" or project_name is None:
        project_name = f"experiment_{generate_random_string()}"

    # generate a random id if not provided
    if id == "":
        id = f"id_{generate_random_string()}"

    print("[INFO] Login to W&B")
    # log in with the W&B API key
    api_key = get_wandb_api_key()
    is_logged_in = wandb.login(key=api_key, relogin=True)

    # initialize W&B run
    run = wandb.init(
        project=project_name,
        # settings=wandb.Settings(start_method="fork"),
        tags=tags,
        save_code=save_code,
        id=id,
        entity="cmu-aie",
        reinit=reinit,
    )
    return run

def capture_hyperparameters(hyperparameters: Dict[str, Any]) -> None:
    """Capture hyperparameters in the current Weights & Biases (W&B) run.

    This function updates the W&B configuration with the provided hyperparameters.

    Args:
        hyperparameters (dict): A dictionary containing hyperparameter names and values.

    Returns:
        None

    Example:
        ```python
        hyperparams = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "num_epochs": 10,
        }
        capture_hyperparameters(hyperparameters)
        ```

    """
    if wandb.run and isinstance(wandb.run, wandb.sdk.wandb_run.Run):
        wandb.config.update(hyperparameters)
    else:
      raise RuntimeError("No Active W&B run. Please ensure you have initialized a W&B run before capturing hyperparameters")

def log_to_wandb(log: Dict[str, Any]) -> None:
    """Log data to the current Weights & Biases (W&B) run.

    This function logs the specified dictionary of data to the W&B run.

    Args:
        log (dict): A dictionary containing data to be logged.

    Returns:
        None

    Raises:
        RuntimeError: If there is no active W&B run.

    Example:
        ```python
        data_to_log = {
            "loss": 0.05,
            "accuracy": 0.95,
        }
        log_to_wandb(data_to_log)
        ```

    """
    if wandb.run and isinstance(wandb.run, wandb.sdk.wandb_run.Run):
        wandb.log(log)
    else:
        raise RuntimeError("No active W&B run. Please ensure you have initialized a W&B run before logging data.")

def close_wandb() -> None:
    """Close the current Weights & Biases (W&B) run.

    This function finishes the active W&B run.

    Returns:
        None

    Raises:
        RuntimeError: If there is no active W&B run.

    Example:
        ```python
        close_wandb()
        ```

    """
    if wandb.run and isinstance(wandb.run, wandb.sdk.wandb_run.Run):
        wandb.finish()
    else:
        raise RuntimeError("No active W&B run. Please ensure you have initialized a W&B run before closing it.")

run_wb_instance = init_wandb()
assert run_wb_instance is wandb.run
close_wandb()
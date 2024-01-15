import os
import wandb
from cryptography.fernet import Fernet
from pathlib import Path
from typing import Dict, List, Any, Optional


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
    try:
      with open(path, 'wb') as _file:
          _file.write(content)
      return path
    except IOError as e:
        print(f"Error saving to file {path}: {e}")

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

def generate_random_character() -> str:
  """"

  """
  import random
  import string
  return random.choices(string.ascii_lowercase)[0]


def generate_random_string(length: int = 10) -> str:
  """Generate a random string of specified length.
  
  Args:
      length (int): Length of the generated string.

  Returns:
      str: A random string of lowercase ASCII characters.
  """
  str = [generate_random_character() for _ in range(length)]
  return "".join(str)

class WandBIntegration:
  """
  
  """
  def __init__(
      self, 
      project_name: str = None, 
      id: str = None,
      config: Dict[str, Any] = None, 
      entity: str = None, 
      save_code: bool = True, 
      reinit: bool = True,
      tags: List[str] = ['baseline']
    ):
      # configure wandb folder
      wandb_folder_path = Path("./config/")
      wandb_folder_path.mkdir(parents=True, exist_ok=True)
      self.fernet_encryption_key_path = wandb_folder_path / "fernet_encryption_key.txt"
      self.wandb_api_path = wandb_folder_path / "wandb_api.txt"
      self.project_name = project_name if project_name else f"experiment_{generate_random_string()}"
      self.id = id if id else f"id_{generate_random_string()}"
      self.config = config if config else {}
      self.entity = entity
      self.save_code = save_code
      self.tags = tags
      self.reinit = reinit,
      self.run = None
      # self.init_run()
  
  def init_run(self):
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

      try:
        api_key = self.get_wandb_api_key()
        is_logged_in = wandb.login(key=api_key, relogin=True)
      except Exception as e:
        raise RuntimeError(f"Error logging in to W&B: {e}")

      try:
        # initialize W&B run
        self.run = wandb.init(
            project=self.project_name,
            # settings=wandb.Settings(start_method="fork"),
            tags=self.tags,
            save_code=self.save_code,
            id=self.id,
            entity=self.entity,
            config=self.config,
            # reinit=self.reinit,
        )

        print("[INFO] W&B run initialized successfully")
      except Exception as e:
        print(f"[ERROR] Error initializing W&B run: {e}")

  def capture_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
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

  def log_metrics(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
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
          wandb.log(metrics, step=step)
      else:
          raise RuntimeError("WandB run is not initialized. Call init_run() before saving metrics.")


  def save_model(self, model_file_path: str):
      if self.run:
          wandb.save(model_file_path)
      else:
          raise RuntimeError("WandB run is not initialized. Call init_run() before saving models.")

  def finish_run(self) -> None:
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
      if self.run:
          wandb.finish()
      else:
          raise RuntimeError("WandB run is not initialized. Call init_run() before finishing the run.")

  def get_and_save_wandb_api_key(
    self,
    api_key: str,
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
      save_to_file(self.wandb_api_path, encrypted_api_key)
      save_to_file(self.fernet_encryption_key_path, fernet_encryption_key)

  def load_and_decrypt_wandb_api_key(
      self,
    ) -> str:
      """Load and decrypt the W&B API key.

      Args:
          wandb_api_path (str): Path to the file where the encrypted API key is stored.
          fernet_encryption_key_path (str): Path to the file where the Fernet key is stored.

      Returns:
          str: The decrypted W&B API key.
      """
      # Load the encrypted API key and the Fernet encryption key
      encrypted_api_key = load_content(self.wandb_api_path)
      fernet_encryption_key = load_content(self.fernet_encryption_key_path)

      # Decrypt and return the API key
      return decrypt_api_key(encrypted_api_key, fernet_encryption_key)

  def get_wandb_api_key(self) -> str:
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
      if os.path.exists(self.wandb_api_path):
        api_key = self.load_and_decrypt_wandb_api_key()
        if api_key:
          # Set the API key in the environment variables
          os.environ['WANDB_API_KEY'] = api_key
          return api_key

      # Ask the user for the API key and save it in the file and environment variable
      api_key = input("Enter your W&B API key: ").strip()
      self.get_and_save_wandb_api_key(api_key)
      os.environ['WANDB_API_KEY'] = api_key
      return api_key
import configparser
import os
from pathlib import Path
from util import EXECUTABLE_DIRECTORY, EXECUTABLE_NAME

config_parser = configparser.ConfigParser()

home = Path.home()
if (XDG_CONFIG_HOME := os.getenv('XDG_CONFIG_HOME', False)):
  config_home = Path(XDG_CONFIG_HOME).expanduser()
else:
  config_home = home / '.config'
config_home /= EXECUTABLE_NAME

config_file = None
config_file_paths = [
  config_home/'config.ini',
  home/f'.{EXECUTABLE_NAME}',
  home/f'.{EXECUTABLE_NAME}/config.ini',
  EXECUTABLE_DIRECTORY/'config.ini'
]

default_args = {
  "iterations": 3,
  "resize": 250,
  "blur_radius": 0.5,
  "minimum_contrast": 50,
}


def read_configuration_from_file():
  global config_file
  _config = {**default_args}  # deep copy

  for config_file_path in config_file_paths:
    if config_file_path.exists():
      config_parser.read(config_file_path)
      config_file = config_file_path.absolute()
      break

  for section in config_parser.sections():
    for option, value in config_parser[section].items():
      option = option.replace('-','_')
      for cast in [int, float]:
        try:
          value = cast(value)
          break
        except ValueError:
          continue
      _config[option] = value

  return _config
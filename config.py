import configparser
import os
from pathlib import Path
from util import EXECUTABLE_DIRECTORY, EXECUTABLE_NAME

config = configparser.ConfigParser()
home = Path("~").expanduser()
config_home = Path(os.getenv('XDG_CONFIG_HOME', "~/.config")).expanduser() / EXECUTABLE_NAME

config_file_locations = [
  config_home/'config.ini',
  home/f'.{EXECUTABLE_NAME}',
  EXECUTABLE_DIRECTORY/'config.ini'
]

default_args = {
  "iterations": 3,
  "resize": 250,
  "blur_radius": 0.5,
  "minimum_contrast": 50,
}

for config_file in config_file_locations:
  if config_file.exists():

    config.read(config_file)

    config_defaults = config['default']
    for option in default_args.keys():
      value = config_defaults.getfloat(option, default_args[option])
      try:
        if value.is_integer(): value = int(value)
        default_args[option] = value
        del config_defaults[option]
      except:
        default_args[option] = value

    kmeans_initial_colors_defaults = config['kmeans-initial-colors']

    default_args |= { k:v for k,v in kmeans_initial_colors_defaults.items() }

    break
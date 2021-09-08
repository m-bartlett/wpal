import argparse
import configparser
import os
from pathlib import Path
from types import SimpleNamespace
from util import popen, EXECUTABLE_DIRECTORY, EXECUTABLE_NAME


parser = argparse.ArgumentParser()

parser.add_argument(
  "--wallpaper-picker",
  "-W",
  type=str,
  default="",
  nargs='?',
  help="launch argument executable as a process and wait for it to exit before generating a wallpaper (intended for interactive wallpaper selectors)"
)

parser.add_argument(
  "--file",
  "-f",
  type=str,
  default="",
  nargs='?',
  help="path to desired image to run palette generation on"
)

parser.add_argument(
  "--light",
  "-l",
  action='store_true',
  help="generate a light theme (the background color is brighter than the foreground colors)"
)

parser.add_argument(
  "--resize",
  "-z",
  type=int,
  help="maximum pixel quantity to resize image to before computing the palette"
)

parser.add_argument(
  "--blur-radius",
  "-b",
  type=float,
  help="radius in pixels to use for box blur before computing the palette"
)

parser.add_argument(
  "--minimum-contrast",
  "-C",
  type=float,
  help="minimum contrast each foreground color should have with the background color"
)

parser.add_argument(
  "--random-color-order",
  "-R",
  default='',
  help="seed for color order",
)

parser.add_argument(
  "--iterations",
  "-i",
  type=int,
  help="kmeans iteration quantity"
)

parser.add_argument(
  "--verbose",
  "-v",
  action='count',
  default=0,
  help="print extra processing information to terminal, supply extra v's for more verbosity"
)

parser.add_argument(
  "--hooks",
  "-x",
  nargs='*',
  # default=True,
  help="execute hook scripts after exporting color information to the environment and Xresources"
)

# parser.add_argument(
#   "--load",
#   "-L",
#   action='store_true',
#   help="load steganographic configuration baked in the image's bytes"
# )

# parser.add_argument(
#   "--save", "-S", action='store_true',
#   help="bake current configuration into the image's bytes"
# )


args = parser.parse_args()


defaults = {
  "iterations": 3,
  "resize": 250,
  "blur_radius": 0.5,
  "minimum_contrast": 50,
}

config = configparser.ConfigParser()
home = Path("~").expanduser()
config_home = Path(os.getenv('XDG_CONFIG_HOME', "~/.config")).expanduser() / EXECUTABLE_NAME

config_file_locations = [
  config_home/'config.ini',
  home/f'.{EXECUTABLE_NAME}',
  EXECUTABLE_DIRECTORY/'config.ini'
]

for config_file in config_file_locations:
  if config_file.exists():
    config.read(config_file)
    config_defaults = config['default']
    for option in defaults.keys():
      value = config_defaults.getfloat(option, defaults[option])
      try:
        if value.is_integer(): value = int(value)
        defaults[option] = value
        del config_defaults[option]
      except:
        defaults[option] = value
    break

for k,v in defaults.items():
  try:
    if getattr(args, k) is None:
      setattr(args, k, v)
  except AttributeError:
    continue
import argparse
from image import ANSI_color_names
from config import default_args


# Configuration priority:
# - CLI args
# - parsed config from serialized image metadata ONLY IF --load is supplied
# - config.ini:
#   - ~/.config/${EXECUTABLE_NAME}/config.ini
#   - ~/.${EXECUTABLE_NAME}
#   - $EXECUTABLE_DIRECTORY/config.ini
# - source-code defaults


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
  "--color-order",
  "-r",
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

parser.add_argument(
  "--load",
  "-L",
  action='store_true',
  help="save given options and palette output for a given image into its metadata "
)

parser.add_argument(
  "--save",
  "-S",
  action='store_true',
  help="parse image metadata for palette configuration"
)

for color_name in ANSI_color_names:
  parser.add_argument(
    f"--{color_name}",
    type=str,
    help=f"specify the kmeans initial cluster color representing {color_name}"
  )


args = parser.parse_args()

# CLI flag values take priority over config
for k,v in default_args.items():
  try:
    if getattr(args, k) is None:
      setattr(args, k, v)
  except AttributeError:
    continue
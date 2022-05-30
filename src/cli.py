import argparse
from .image import ANSI_color_names
from .config import read_configuration_from_file


# Configuration priority:
# - CLI args
# - parsed config from serialized image metadata ONLY IF --load is supplied
# - config.ini:
#   - ~/.config/${EXECUTABLE_NAME}/config.ini
#   - ~/.${EXECUTABLE_NAME}
#   - ~/.${EXECUTABLE_NAME}/config.ini
#   - $EXECUTABLE_DIRECTORY/config.ini
# - source-code defaults


parser = argparse.ArgumentParser()

parser.add_argument(
  "--wallpaper-picker",
  "-W",
  type=str,
  default=False,
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
  default=False,
  nargs='?',
  help="""
  if a string representing a permutation of a subset all non-black and non-white ANSI color indices 123456 (e.g. 6142 or 512346) is given, it will first be completed with any remaining indices in sorted order (e.g. 6142 -> 614235, 512346 -> 512346) and then will be used as the index ordering for the "sorted" color variables in the output (e.g. 1->6, 2->1, 3->4, 4->2, 5->3, 6->5). This does not affect the ANSI color order (e.g. 31 and 41 will still be the palette's red), this only affects the color variables explicitly named resorted in the output (which are intended for use in other application palettes to help create a unique look for each wallpaper). If the argument provided cannot be interpreted as a permutation of 123456 (e.g. 61422) then the argument will be hashed and then treated as a seed for a psuedo-random color ordering (e.g. 61422 -> randomsort(123456, seed=hash(61422)) -> 345216). If no argument is given with this flag, then an entirely random color palette will be used, and the resulting index ordering will be printed when using high verbosity (-vvv) so the user may have access to the index order of a sort they wish to preserve, e.g. with --save.
  """
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

parser.add_argument(
  "--pure",
  "-P",
  action='store_true',
  help="use pure 3-bit ANSI colors as k-means starting points (results in highest contrast but usually produces over-satured palettes)"
)

for color_name in ANSI_color_names:
  parser.add_argument(f"--{color_name}",
                      type=str,
                      help=f"specify the kmeans initial cluster color representing {color_name}")

for color_index in range(0,16):
  parser.add_argument(f"-{color_index}",
                      type=str,
                      dest=str(color_index),
                      help=f"Override ANSI color {color_index} with a given color")


args = parser.parse_args()



# CLI flag values take priority over configuration from files
file_config = read_configuration_from_file()
for k,v in file_config.items():
  try:
    if getattr(args, k, None) is None:
      setattr(args, k, v)
  except AttributeError:
    continue
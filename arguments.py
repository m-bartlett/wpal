import argparse


defaults = {
  "iterations": 3,
  "resize": 250,
  "blur_radius": 0.5,
  "minimum_contrast": 50,
  # light_minimum_contrast: 50,
  # dark_minimum_contrast: 30
}


parser = argparse.ArgumentParser()

parser.add_argument(
  "--wallpaper-picker",
  "-W",
  type=str,
  help="launch argument executable as a process and wait for it to exit before generating a wallpaper (intended for interactive wallpaper selectors)"
)

parser.add_argument(
  "--file",
  "-f",
  type=str,
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

# parser.add_argument(
#   "--no-op",
#   "-0",
#   action='store_true',
#   help="a no-op to use defaults without changing background image"
# )

parser.add_argument(
  "--verbose",
  "-v",
  action='count',
  default=0,
  help="print extra processing information to terminal, supply extra v's for more verbosity"
)

parser.add_argument(
  "--hooks",
  "-H",
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

for k,v in defaults.items():
  try:
    if getattr(args, k) is None:
      setattr(args, k, v)
  except AttributeError:
    continue
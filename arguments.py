import argparse

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
  help="generate light theme instead"
)

parser.add_argument(
  "--resize",
  "-z",
  type=int,
  help="pixel quantity to resize image to before calculating palette"
)

parser.add_argument(
  "--blur-radius",
  "-b",
  type=float,
  help="radius in pixels to use for box blur before kmeans clustering"
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
  "--no-op",
  "-0",
  action='store_true',
  help="a no-op to use defaults without changing background image"
)

parser.add_argument(
  "--verbose",
  "-v",
  action='count',
  default=0,
  help="print extra palette coloration to terminal"
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


args.iterations = args.iterations or 3
args.resize = args.resize or 250
args.blur_radius = args.blur_radius or 0.5
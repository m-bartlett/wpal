#!/usr/bin/python3
import numpy as np
import sys
import argparse
import subprocess
import shlex
import pathlib
import os

from core import generate_ANSI_palette_from_pixels
from image import *
from cli import args
from util import EXECUTABLE_DIRECTORY

VERBOSE_DEBUG  = args.verbose > 3
VERBOSE_HIGH   = VERBOSE_DEBUG or args.verbose > 2
VERBOSE_MEDIUM = VERBOSE_HIGH or args.verbose > 1
VERBOSE_LOW    = VERBOSE_MEDIUM or args.verbose > 0

if args.hooks is None and args.verbose == 0:
  info("No execution hooks or output verbosity specified, executing default hooks.")
  args.hooks=[]

np.set_printoptions(precision=3, suppress=True)


if VERBOSE_HIGH:
  from config import config_file
  from cli import file_config
  if config_file:
    info(f"Configuration loaded from {config_file}:")
    for key, value in file_config.items():
      info(f"\t{key}={value}")
  info("\nArguments: " + ' '.join([f"{k}={v}" for k,v in args.__dict__.items()]))


if args.wallpaper_picker is not False:
  if args.wallpaper_picker is None:
    popen(args.wallpaper_pick_command)
  else:
    popen(args.wallpaper_picker)

if args.file:
  wallpaper_path = args.file
elif args.wallpaper_path:
  wallpaper_path = args.wallpaper_path
elif args.wallpaper_path_command:
  wallpaper_path = popen(args.wallpaper_path_command).stdout.decode().strip()
else:
  raise RuntimeError("A wallpaper path was not provided")

if VERBOSE_MEDIUM:  info(f"\nUsing wallpaper: {wallpaper_path}")


if args.load:
  from exif import load_exif_metadata
  Xresource_colors = load_exif_metadata(wallpaper_path)
  if VERBOSE_LOW:
    info("Found palette cache embedded in image metadata")
    info(Xresource_colors['config'])
  del Xresource_colors['config']



kmeans_initial_colors = ANSI.copy()
if not args.pure:
  for i, color_name in enumerate(ANSI_color_names):
    try:
      c = string2rgb(getattr(args, color_name))
      if len(c) > 0: kmeans_initial_colors[i] = c
    except AttributeError:
      continue


wallpaper = Image.open(wallpaper_path).convert('RGB')

if args.resize > 0:
  wallpaper.thumbnail((args.resize, args.resize), resample=Image.LANCZOS)
if args.blur_radius:
  wallpaper = wallpaper.filter(ImageFilter.BoxBlur(radius=args.blur_radius))

rgb_pixels = np.array(wallpaper, dtype=int)[:,:,:3]
rgb_pixels = rgb_pixels.reshape( rgb_pixels.shape[0] * rgb_pixels.shape[1],
                                 rgb_pixels.shape[2] )


palette_result = (

    generate_ANSI_palette_from_pixels(
      rgb_pixels            = rgb_pixels,
      kmeans_initial_colors = kmeans_initial_colors,
      kmeans_iterations     = args.iterations,
      minimum_contrast      = args.minimum_contrast,
      light_palette         = args.light,
      verbose               = VERBOSE_HIGH
    )

)

base_colors, bold_colors = palette_result["base_colors"], palette_result["bold_colors"]
highlight,   lowlight    = palette_result["highlight"],   palette_result["lowlight"]
ansi_palette = np.concatenate([base_colors, bold_colors])

midground = (0.8*base_colors[0] + 0.2*bold_colors[7]).astype(np.uint8)
midground = ( (midground_weight:=0.7) * base_colors[0] +
              (1-midground_weight) * bold_colors[7] ).astype(np.uint8)



if args.color_order is None:  # -r is given with no argument, use a random ordering
  color_order = parse_string_as_color_order_or_random_seed(os.urandom(6))
  info(f"Using palette ordering: {''.join(map(str,color_order))}")
elif args.color_order is not False:  # -r is given with an argument, parse if its a seed or completable ordering
  color_order = parse_string_as_color_order_or_random_seed(args.color_order)
else:   # -r is absent, use the kmeans weighted neighbor order
  color_order = palette_result["weighted_order"]


sorted_base_colors = base_colors[color_order]
sorted_bold_colors = bold_colors[color_order]


# Map Xresource color name to RGB tuple
Xresource_colors = {
  **{ f"color{i}"   : c for i,c in enumerate(ansi_palette) },
  # **{ f"color{i}D"  : c for i,c in enumerate(palettes[0]) },
  # **{ f"color{i}d"  : c for i,c in enumerate(palettes[1]) },
  # **{ f"color{i}l"  : c for i,c in enumerate(palettes[3]) },
  # **{ f"color{i}L"  : c for i,c in enumerate(palettes[4]) },
  **{ f"color{i+1}s": c for i,c in enumerate(sorted_base_colors) },
  **{ f"color{i+1}S": c for i,c in enumerate(sorted_bold_colors) },
  "background": ansi_palette[0],
  "midground":  midground,
  "foreground": ansi_palette[15],
  "highlight":  highlight,
  "lowlight":   lowlight,
}

# Convert all RGB tuples to hexidecimal strings
Xresource_colors = { color: rgb2hex(rgb) for color, rgb in Xresource_colors.items() }
Xresource_colors["themestyle"] = 'light' if args.light else 'dark'


if args.save:
  from exif import save_exif_metadata
  if not save_exif_metadata(wallpaper_path, args, Xresource_colors):
    info("An error occurred saving palette cache to image metadata.")


if VERBOSE_DEBUG:
  info()
  for color_name, color_hex in Xresource_colors.items():
    print(f"{color_name}={color_hex}")


if VERBOSE_MEDIUM:  # Show in-terminal image preview at higher verbosities

  with TerminalImagePreview(wallpaper, padding=(1,1,2,1)) as preview:
    from time import sleep

    if VERBOSE_DEBUG:

      for palette_batch in [ [kmeans_initial_colors],
                             [base_colors,bold_colors]+[[highlight, lowlight]],
                             list(palettes),
                             [sorted_base_colors, sorted_bold_colors] ]:
        for palette in palette_batch:
          info(palette_as_colorized_hexcodes(palette, separator=" "))
        info()

    else:
      preview.display_image()
      sleep(0.2)

      print_palette_preview( base_colors=base_colors,
                             bold_colors=bold_colors,
                             highlight=highlight,
                             lowlight=lowlight )


    _input = sys.stdin.read(1)
    if _input in ["q", "\x1b"]: # If ESC or q is pressed, exit failure
      sys.exit(1)


if VERBOSE_LOW:
  info()
  pretty_print_palette( base_colors=base_colors,
                        bold_colors=bold_colors,
                        highlight=highlight,
                        lowlight=lowlight )


if args.hooks is not None:
  if len(args.hooks) == 0:
    hooks = list(map(str, (EXECUTABLE_DIRECTORY/'hooks').glob('*')))
  else:
    hooks = [hook for hook in args.hooks if os.path.exists(hook)]

  from Xresources import xrdb_merge
  p = xrdb_merge(Xresource_colors)
  if p.returncode != 0:
    raise RuntimeError(f"xrdb merge did not exit successfully: {p.stderr}")


  # Export colors as hexidemical to the environment
  os.environ |= Xresource_colors

  if VERBOSE_DEBUG:
    info(f"\nExecuting hooks:\n{chr(10).join(hooks)}")


  # Execute all hooks concurrently in their own thread
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor() as executor:
    processes = list(executor.map(popen, hooks))
    for p in processes:
      if VERBOSE_HIGH:
        info(f"Executed {p.args[0]}")
      if p.returncode != 0:
        info(f"WARNING: {p.args[0]} returned nonzero exit code.")
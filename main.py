#!/usr/bin/python3
import numpy as np
import sys
import argparse
import subprocess
import shlex
import pathlib
import os

from image import *
from options import args, config, config_file, config_defaults
from kmeans import kmeans
from util import EXECUTABLE_DIRECTORY

VERBOSE_DEBUG  = args.verbose > 3
VERBOSE_HIGH   = VERBOSE_DEBUG or args.verbose > 2
VERBOSE_MEDIUM = VERBOSE_HIGH or args.verbose > 1
VERBOSE_LOW    = VERBOSE_MEDIUM or args.verbose > 0

np.set_printoptions(precision=3, suppress=True)

if VERBOSE_HIGH:
  if config_file:
    info(f"Configuration loaded from {config_file}:")
    for key, value in config_defaults.items():
      info(f"\t{key}={value}")
  info("\nArguments: " + ' '.join([f"{k}={v}" for k,v in args.__dict__.items()]))

if args.wallpaper_picker:
  popen(args.wallpaper_picker)

if args.file:
  wp_path = args.file
else:
  wp_path = get_current_wallpaper()

if VERBOSE_MEDIUM:
  info(f"\nUsing wallpaper: {wp_path}")

wp=Image.open(wp_path).convert('RGB')
wp.thumbnail((args.resize, args.resize), resample=Image.LANCZOS)
if args.blur_radius:
  wp = wp.filter(ImageFilter.BoxBlur(radius=args.blur_radius))

pixels=np.array(wp, dtype=int)[:,:,:3]
pixels=pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2] )

pixels = filter_colors_in_ellipsoid_volume(
  pixels,
  ellipsoids = [
      {   # Filter brown
          "radii":  np.array([100,100,150]),
          "offset": np.array([128, 128, 0])
      },
  ]
)

(
  kmeans_cluster_centers, ANSI_indices_sorted_by_neighbor_quantity
)   = kmeans(pixels, args.iterations)

initial_palette = (
  constrain_background_colors_to_minimum_distance_from_target(
    kmeans_cluster_centers,
    constraints = [
      {"color":0, "max_distance":10000},
      {"color":7, "max_distance":5000},
    ]
  )
)

if args.light:
  initial_palette[[0,7]] = initial_palette[[7,0]] # Swap white and black, i.e. fg with bg


if args.minimum_contrast:
  foreground_colors = initial_palette[1:-1]
  background_color  = initial_palette[0]
  higher_contrast_foreground_colors = (
    constrain_contrast_between_foreground_and_background_colors(
      foreground_colors = foreground_colors,
      background_color = background_color,
      minimum_contrast = args.minimum_contrast,
      verbose = VERBOSE_HIGH
    )
  )
  initial_palette[1:-1] = higher_contrast_foreground_colors


hsv_palette = rgb_palette_to_hsv_palette(initial_palette)

highlight_index = get_most_saturated_color_index(hsv_palette)

value_scalars = [0.9, 0.95, 1.25, 1.5]
saturation_scalars = (
  [1.5, 1.25, 1.0, 0.95] if args.light
  else [1, 1, 1, 1]
)

palettes = (
  create_gradated_palettes(
    hsv_palette,
    value_scalars = value_scalars,
    saturation_scalars = saturation_scalars
  )
)

middle_palette_index = (middle_palette_index:=len(palettes))//2 + (middle_palette_index & 1)


if args.random_color_order:
  # Sort using deterministic psuedo-random ordering based on file hash

  import hashlib
  # seed=int(hashlib.sha256(open(wp_path,'rb').read()).hexdigest(), 16) % 4294967295
  seed=int(hashlib.sha256(args.random_color_order.encode()).hexdigest(), 16) % 4294967295
  color_order = np.arange(1,ANSI.shape[0]-1)
  np.random.seed(seed)
  np.random.shuffle(color_order)

else:
  # Sort by cluster sizes descending from k-means
  color_order = ANSI_indices_sorted_by_neighbor_quantity


if args.light:
  bold_colors = palettes[1]
  base_colors = palettes[-1].copy()
  base_colors[0] = palettes[-1][0]
  highlight = bold_colors[highlight_index]
  lowlight = palettes[-3][highlight_index]

else:
  base_colors = palettes[2]
  bold_colors = palettes[-1]
  highlight = bold_colors[highlight_index]
  lowlight = palettes[1][highlight_index]


ansi_palette = np.concatenate([base_colors, bold_colors])

midground = (0.7*base_colors[0] + 0.3*bold_colors[7]).astype(np.uint8)

sorted_base_colors = base_colors[color_order]
sorted_bold_colors = bold_colors[color_order]


# Map Xresource color name to RGB tuple
Xresource_colors = {
  **{ f"color{i}"   : c for i,c in enumerate(ansi_palette) },
  **{ f"color{i}D"  : c for i,c in enumerate(palettes[0]) },
  **{ f"color{i}d"  : c for i,c in enumerate(palettes[1]) },
  **{ f"color{i}l"  : c for i,c in enumerate(palettes[3]) },
  **{ f"color{i}L"  : c for i,c in enumerate(palettes[4]) },
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


if VERBOSE_DEBUG:
  info()
  for color_name, color_hex in Xresource_colors.items():
    print(f"{color_name}={color_hex}")


if VERBOSE_MEDIUM:  # Show in-terminal image preview at higher verbosities

  with TerminalImagePreview(wp) as preview:
    from time import sleep
    sleep(0.05)

    if VERBOSE_DEBUG:

      for palette_batch in [ [ANSI],
                             [base_colors,bold_colors]+[[highlight, lowlight]],
                             list(palettes),
                             [sorted_base_colors, sorted_bold_colors] ]:
        for palette in palette_batch:
          info(palette_as_colorized_hexcodes(palette, separator=" "))
        info()

    else:

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
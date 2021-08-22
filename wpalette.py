#!/usr/bin/python3
import numpy as np
import sys
import argparse
import subprocess
import shlex
import pathlib
import os
import tempfile

from image import *
from arguments import args
from kmeans import *

np.set_printoptions(precision=3, suppress=True)

if args.wallpaper_picker:
  popen(args.wallpaper_picker)

if args.file:
  wp_path = args.file
else:
  wp_path = get_current_wallpaper()

if args.verbose > 1:
  printerr(f"Using wallpaper: {wp_path}\n")

wp=Image.open(wp_path).convert('RGB')
wp.thumbnail((args.resize, args.resize), resample=Image.LANCZOS)
if args.blur_radius:
  wp = wp.filter(ImageFilter.BoxBlur(radius=args.blur_radius))

pixels=np.array(wp, dtype=int)[:,:,:3]
pixels=pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2] )

pixels = (
  filter_colors_in_ellipsoid_volume(
    pixels,
    ellipsoids = [
        {   # Filter brown
            "radii":  np.array([100,100,150]),
            "offset": np.array([128, 128, 0])
        },
    ]
  )
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
      verbose = args.verbose > 2
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


# Sort using deterministic psuedo-random ordering based on file hash
if args.random_color_order:
  import hashlib
  # seed=int(hashlib.sha256(open(wp_path,'rb').read()).hexdigest(), 16) % 4294967295
  seed=int(hashlib.sha256(args.random_color_order.encode()).hexdigest(), 16) % 4294967295
  color_order = np.arange(1,ANSI.shape[0]-1)
  np.random.seed(seed)
  np.random.shuffle(color_order)

else:
  # sort by cluster sizes descending from k-means
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


# Print shell-declarable strings to stdout for consumption
for n,c in {
        **{ f"color{i}"   : c for i,c in enumerate(ansi_palette) },
        **{ f"color{i}D"  : c for i,c in enumerate(palettes[0]) },
        **{ f"color{i}d"  : c for i,c in enumerate(palettes[1]) },
        **{ f"color{i}l"  : c for i,c in enumerate(palettes[3]) },
        **{ f"color{i}L"  : c for i,c in enumerate(palettes[4]) },
        **{ f"color{i+1}s": c for i,c in enumerate(sorted_base_colors) },
        **{ f"color{i+1}S": c for i,c in enumerate(sorted_bold_colors) },
        "highlight":highlight,
        "lowlight":lowlight,
        "midground":midground,
        "highlight_fg":most_visible_foreground_color(highlight),
      }.items():

  print("{0}=#{1:02X}{2:02X}{3:02X}".format(n, *c))


if args.verbose > 1:  # Show in-terminal image preview at high verbosity

  with TerminalImagePreview(wp) as preview:

    if args.verbose > 2:

      verbose_palettes = (
        [base_colors,bold_colors] +
        [[highlight, lowlight]] +
        [ANSI] +
        list(palettes) +
        [sorted_base_colors, sorted_bold_colors]
      )

      palette_printer = lambda palette: print_palette_as_colorized_hexcodes(palette)

    else:
      # printerr(f"\x1b[{preview.TERM_HEIGHT//2};{preview.TERM_WIDTH//2}H", end='')
      verbose_palettes = [base_colors,bold_colors] + [[highlight, lowlight]]
      palette_printer = lambda palette: print_palette_as_filled_blocks(palette, block_content="     ")


    for palette in verbose_palettes:
      palette_printer(palette)

    if sys.stdin.read(1) == "\x1b": # If ESC is pressed, exit failure
      sys.exit(1)


if args.verbose > 0:
	DELIMITER="   "
	for line in zip([base_colors,bold_colors], [[lowlight],[highlight]]):
	  printerr(
	    " ".join(
	      [
	        "".join(
	          [
	            "\x1b[48;2;{0};{1};{2}m{3}\x1b[0m".format(*c, DELIMITER)
	            for c in part
	          ]
	        )
	        for part in line
	      ]
	    )
	  )

# avg_color = pixels.mean(axis=0).astype(int)

# avg_closest_color = accents[np.argmin( ( ( accents[1:-1] - ANSI[1:-1] )**2 ).sum(axis=1) ) + 1]
# printerr(
#   "".join(
#     [
#           "\x1b[48;2;{0};{1};{2}m{3}\x1b[0m".format(*c, DELIMITER)
#           for c in [avg_color, avg_closest_color]
#     ]
#   )
# )

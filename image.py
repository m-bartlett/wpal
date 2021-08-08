import numpy as np
import contextlib
from util import *
from PIL import Image, ImageFilter
from colorsys import hsv_to_rgb, rgb_to_hsv

BLACK=np.array([0,0,0])
WHITE=np.array([255,255,255])

ANSI = np.uint8(
  [
    list(map(int, c.partition('(')[2][:-1].split(',')))
    for c in [

      # "rgb(0,40,40)",
      # "rgb(40,40,40)",
      "rgb(15,15,15)",
      "rgb(220,75,75)",
      "rgb(75,220,75)",
      "rgb(220,180,75)",
      "rgb(75,100,220)",
      "rgb(120,75,220)",
      "rgb(75,170,220)",
      "rgb(220,220,220)"

      # # Muted
      # "rgb(0,40,40)",
      # "rgb(148,0,9)",
      # "rgb(00,180,80)",
      # "rgb(255,120,0)",
      # "rgb(80,120,220)",
      # "rgb(140,82,162)",
      # "rgb(90, 165, 185)",
      # "rgb(200,200,200)"

      # # Hybrid
      # "rgb(15,15,15)",
      # # "rgb(188,0,9)",
      # "rgb(255,0,0)",
      # "rgb(50,255,50)",
      # "rgb(255,120,0)",
      # "rgb(0,100,255)",
      # "rgb(140,0,255)",
      # "rgb(0,170,200)",
      # "rgb(215,215,215)"

    ]
  ]
)


def luminance(pixel):
  rgb = pixel / 255
  l=[c/ 12.92 if c <= 0.03928 else ((c + 0.0055) / 1.055) ** 2.4 for c in rgb]
  return l[0] * 0.2126 + l[1] * 0.7152 + l[2] * 0.0722;


def contrast(rgb1, rgb2):
  return (luminance(rgb1)+ 0.05) / (luminance(rgb2) + 0.05);


def most_visible_foreground_color(rgb):
  global WHITE
  global BLACK
  if contrast(rgb,WHITE) < contrast(BLACK,rgb):
    return WHITE
  else:
    return BLACK


def filter_colors_in_ellipsoid_volume(pixels, ellipsoids=[]):
  for ellipsoid in ellipsoids:
      pixels = pixels[
          (  ( (pixels - ellipsoid['offset']) / ellipsoid['radii'] )**2  ).sum(axis=1) > 1
      ]
  return pixels


def constrain_background_colors_to_minimum_distance_from_target(palette, constraints=[]):
  for constraint in constraints:
    color = constraint['color']
    constraint_distance = constraint['max_distance']
    for step in range(10):
      distance_from_target_color = ( (palette[color] - ANSI[color])**2 ).sum()
      if distance_from_target_color < constraint_distance: break
      palette[c] = ((0.9*palette[color]) + (0.1*ANSI[color]))
    return palette


def rgb_palette_to_hsv_palette(palette):
  return np.apply_along_axis(lambda c: rgb_to_hsv(*c), 1, palette)


def get_most_saturated_color_index(hsv_palette):
  # saturation_distances = np.sqrt( (hsv_palette[1:-1,1] - 1.0 )**2 + ((hsv_palette[1:-1,2]/255) - 1.0 )**2 )
  saturation_distances = (hsv_palette[1:-1,1] - 1.0)**2
  # saturation_distances = (hsv_palette[1:-1,2] - 255)**2
  return saturation_distances.argmin()+1


def create_gradated_palettes(hsv_palette, value_scalars, saturation_scalars):
  # Create an array that is several duplicates of the original palette with different values
  palettes = np.tile(hsv_palette, [len(value_scalars)+1, 1, 1])

  print(palettes)

  for i, value_scalar in enumerate(value_scalars):
    palette = palettes[i]

    # 0=hue, 1=saturation, 2=value, we want to change value
    palette[:,2] = np.minimum(
      palette[:,2] * value_scalar,
      np.repeat(255,palette.shape[0])
    )

    palette[1:-1,1] = np.minimum(
      palette[1:-1,1] * saturation_scalars[i],
      np.repeat(1.0,palette.shape[0]-2)
    )

  palettes = np.apply_along_axis(lambda c: hsv_to_rgb(*c), 2, palettes).astype(int)

  # place original palette in center, making them value-sorted
  palettes_len = len(palettes)
  middle_index = sum(divmod(palettes_len,2))-1
  palette_order = list(range(palettes_len))
  palette_order.insert(middle_index, palette_order.pop())
  palettes = palettes[palette_order]

  return palettes




class AlternateBufferManager(contextlib.AbstractContextManager):
  def __enter__(self):
      printe("\033[?1049h\033[0H\033[2J") # switch to secondary buffer and clear it
  def __exit__(self, exc_type, exc_value, exc_traceback):
      printe(f"\033[2J\033[?1049l")       # switch back to primary buffer


def terminal_image_preview(image):
  # Print image preview to terminal using w3m
  # https://blog.z3bra.org/2014/01/images-in-terminal.html

  import tempfile

  WIDTH_SCALAR = 7
  HEIGHT_SCALAR = 18
  WIDTH, HEIGHT = get_terminal_size()
  WIDTH, HEIGHT = WIDTH*WIDTH_SCALAR , HEIGHT*HEIGHT_SCALAR
  W3M_IMGDISPLAY_BIN = "/usr/lib/w3m/w3mimgdisplay"


  with AlternateBufferManager():

    with tempfile.NamedTemporaryFile(suffix=f'.jpg') as tf:
      image.save(tf.name)
      popen(
        W3M_IMGDISPLAY_BIN,
        stdin=f"0;1;0;0;{WIDTH};{HEIGHT};;;;;{tf.name}\n4;\n3;\n"
      )

  input()

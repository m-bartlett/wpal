import numpy as np
import contextlib
import tempfile
import termios
import re
from util import *
from PIL import Image, ImageFilter
from colorsys import hsv_to_rgb, rgb_to_hsv
from hashlib import sha256


np.set_printoptions(precision=3, suppress=True)

RGB_STRING_REGEX = re.compile(r'rgba?\((\d+),(\d+),(\d+)(?:,\d+)?\)')
HEX_STRING_REGEX = re.compile(r'(?:0[xX]|#)?([0-9a-fA-F]{3,8})')

BLACK=np.array([0,0,0])
WHITE=np.array([255,255,255])
ANSI_color_names = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
ANSI = np.uint8([
  [0,   0,   0  ],
  [255, 0,   0  ],
  [0,   255, 0  ],
  [255, 255, 0  ],
  [0,   0,   255],
  [255, 0,   255],
  [0,   255, 255],
  [255, 255, 255]
])


def luminance(pixel):
  try:
    rgb = pixel / 255
  except TypeError:
    rgb = np.array(pixel) / 255
  l=[c/ 12.92 if c <= 0.03928 else ((c + 0.0055) / 1.055) ** 2.4 for c in rgb]
  return l[0] * 0.2126 + l[1] * 0.7152 + l[2] * 0.0722;


def luminant_contrast(rgb1, rgb2):
  return (luminance(rgb1)+ 0.05) / (luminance(rgb2) + 0.05);


def contrast(rgb1, rgb2):
  rgb1 = np.array(rgb1).astype(float)
  rgb2 = np.array(rgb2).astype(float)
  return abs(  ( (rgb2.min() + rgb2.max()) - (rgb1.min() + rgb1.max()) ) / 2  )


def most_visible_foreground_color(rgb, white=WHITE, black=BLACK):
  if luminant_contrast(rgb,white) > luminant_contrast(black,rgb):
    return black
  else:
    return white


def validate_rgb_palette(palette):
  if isinstance(palette, np.ndarray):
    palette = np.minimum(palette, np.full(palette.shape, 255, dtype=np.uint8))
    palette = np.maximum(palette, np.zeros(palette.shape, dtype=np.uint8))
    palette = palette.astype(np.uint8)
  else:
    palette = [ [ min(max(c,0),255) for c in rgb] for rgb in palette ]
  return palette


def rgb2hex(rgb):
  return "#{0:02X}{1:02X}{2:02X}".format(*(round(c) for c in rgb))


def rgb_string2rgb(s):
  try:
    return np.uint8(RGB_STRING_REGEX.findall(s)[0])
  except IndexError:
    return ''


def hex2rgb(s):
  h = HEX_STRING_REGEX.findall(s)
  if not h: return ''
  h = h[0]
  if len(h) > 5:
    r, g, b = h[0:2], h[2:4], h[4:6]
  else:
    r, g, b = h[0], h[1], h[2]
    r, g, b = f'{r}{r}', f'{g}{g}', f'{b}{b}'
  return np.uint8(list(map(lambda x: int(x, 16), [r,g,b])))


def string2rgb(s):
  rgb = rgb_string2rgb(s)
  if len(rgb) == 0:
    return hex2rgb(s)
  else:
    return rgb


def ANSI_colorize(message, fg='', bg=''):
  if isinstance(fg, np.ndarray):
    fg = fg.astype(np.uint8)
  fg = "38;2;{0};{1};{2}".format(*fg)
  if isinstance(bg, np.ndarray):
    bg = bg.astype(np.uint8)
  bg = "48;2;{0};{1};{2}".format(*bg)
  return f"\x1b[{bg};{fg}m{message}\x1b[0m"


def rgb2ANSI_colorized_hex(rgb):
  return ANSI_colorize(
    rgb2hex(rgb),
    fg=most_visible_foreground_color(rgb),
    bg=rgb
  )


def palette_as_colorized_hexcodes(palette, separator=""):
  palette = validate_rgb_palette(palette)
  return separator.join([rgb2ANSI_colorized_hex(rgb) for rgb in palette])


def palette_as_foreground_on_background_ANSI_colors( foreground_colors,
                                                     background_color,
                                                     separator=""      ):
  foreground_colors = validate_rgb_palette(foreground_colors)
  background_color = validate_rgb_palette(background_color)
  return separator.join([
    ANSI_colorize(rgb2hex(rgb), fg=rgb, bg=background_color)
    for rgb in foreground_colors
  ])


def palette_as_filled_blocks(palette, block_content=" ", separator=""):
  palette = validate_rgb_palette(palette)
  return separator.join([
    ANSI_colorize(block_content, fg=rgb, bg=rgb)
    for rgb in palette
  ])


def pretty_print_palette( *,
                          base_colors,
                          bold_colors,
                          highlight,
                          lowlight,
                          block_content="   ",
                          palette_separator="",
                          highlight_separator="  " ):
  for line in zip([base_colors,bold_colors], [[lowlight],[highlight]]):
    info(
      highlight_separator.join([
        palette_as_filled_blocks(group, block_content=block_content, separator=palette_separator)
        for group in line
      ])
    )


def print_palette_preview(*, base_colors, bold_colors, highlight, lowlight):
  bg = base_colors[0]
  fg = bold_colors[7]
  spacer_width = len("#000000")
  spacer = " " * spacer_width
  palette_info_width = spacer_width * 6 + 6
  height, width = get_terminal_size()
  offset_width = (width - palette_info_width) // 2
  offset = f"\033[{offset_width}C"
  # space = "\033[1C"
  space = " "
  info(
    offset +
    ANSI_colorize(space + spacer + rgb2hex(bg) + spacer + space, fg=highlight, bg=bg) + space +
    ANSI_colorize(space + spacer + rgb2hex(fg) + spacer + space, fg=lowlight, bg=fg)
  )
  info()

  base_foreground_colors = base_colors[1:-1].copy()[[0,2,1,5,3,4]]
  info(offset + palette_as_foreground_on_background_ANSI_colors(base_foreground_colors, bg, separator=space))
  info(offset + palette_as_filled_blocks(base_foreground_colors, block_content=spacer, separator=space))

  bold_foreground_colors = bold_colors[1:-1].copy()[[0,2,1,5,3,4]]
  info(offset + palette_as_filled_blocks(bold_foreground_colors, block_content=spacer, separator=space))
  info(offset + palette_as_foreground_on_background_ANSI_colors(bold_foreground_colors, bg, separator=space))

  # codeblock_json_keys = ['"red":   ', '"yellow":', '"green": ', '"cyan":  ', '"blue":  ', '"purple":' ]
  # codeblock_json_key_width = (max(map(len, codeblock_json_keys)) + spacer_width) // 2 + 3
  # codeblock_offset = ' ' * codeblock_json_key_width
  # codeblock_offset_colored = ANSI_colorize(codeblock_offset, bg=bg, fg=fg)

  # info()
  # info(offset + ANSI_colorize("{" + (" "*46), fg=fg, bg=bg))
  # for i, key in enumerate(codeblock_json_keys):
  #   key = ANSI_colorize(key, fg=base_foreground_colors[i], bg=bg)
  #   bold_value = bold_foreground_colors[i]
  #   value = ANSI_colorize(rgb2hex(bold_value), fg=bold_value, bg=bg)
  #   line_string = f'{key} "{value}",'
  #   info(offset + codeblock_offset_colored + line_string)
  # info(offset + ANSI_colorize("}" + (" "*46), fg=fg, bg=bg))


def parse_string_as_color_order_or_random_seed(order):
    """
    Check if arg is a subset of elements 1-6, and if so
    return the subset plus remaining elements in order.
    Otherwise, use arg to seed a psuedo-random number
    generator and use it to shuffle elements 1-6.
    This list will serve as the index order for ANSI
    colors red, green, yellow, blue, cyan, and violet.
    """
    try:
      order = [int(n) for n in order]
      target_order = set(range(1,7))
      order_difference = target_order.difference(order)
      completed_order = list(order) + list(order_difference)
      if len(order) + len(order_difference) == len(target_order):
        return completed_order
      else:
        raise ValueError
    except ValueError:
      import pickle
      order = pickle.dumps(order)
      seed = int(sha256(bytes(order)).hexdigest(), 16) % 4294967295
      color_order = list(target_order)
      np.random.seed(seed)
      np.random.shuffle(color_order)
      return color_order


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
      palette[color] = ((0.9*palette[color]) + (0.1*ANSI[color]))
    return palette


def rgb_palette_to_hsv_palette(palette):
  return np.apply_along_axis(lambda c: rgb_to_hsv(*c), 1, palette)


def get_most_saturated_color_index(hsv_palette):
  # saturation_distances = np.sqrt( (hsv_palette[1:-1,1] - 1.0 )**2 + ((hsv_palette[1:-1,2]/255) - 1.0 )**2 )
  saturation_distances = (hsv_palette[1:-1,1] - 1.0)**2
  # saturation_distances = (hsv_palette[1:-1,2] - 255)**2
  return saturation_distances.argmin()+1


def rebalance_palette(hsv_palette, value, saturation):
  palette = hsv_palette.copy()

  # 0=hue, 1=saturation, 2=value
  # hue<=360, saturation<=1, value<=255
  palette[:,2] = np.minimum( palette[:,2] * value,
                             np.repeat(255,palette.shape[0]) )

  # don't re-saturate background and foreground color since they're superlative
  palette[1:-1,1] = np.minimum( palette[1:-1,1] * saturation,
                                np.repeat(1.0,palette.shape[0]-2) )

  palette = np.apply_along_axis(lambda c: hsv_to_rgb(*c), 1, palette).astype(np.uint8)

  return palette


def constrain_contrast_between_foreground_and_background_colors(
      *,
      foreground_colors, # Assumes a uint8 numpy array with shape (6,3)
      background_color,  # Assumes a uint8 numpy array with shape (1,3)
      minimum_contrast=30,
      minimum_error=0.1,
      max_iterations=60,  # convergence isn't guaranteed, prevent infinite loop
      verbose=False,
    ):

  deltas = foreground_colors - background_color
  magnitudes = np.linalg.norm(deltas, axis=1)
  gradients = deltas / magnitudes[:, np.newaxis]

  light_background = background_color.mean() > foreground_colors.mean()

  if light_background:
    _contrast = lambda color: contrast(background_color, color)
  else:
    _contrast = lambda color: contrast(color, background_color)

  contrasts = np.apply_along_axis(_contrast, axis=1, arr=foreground_colors)

  # contrast function isn't affine proportional, but this is a decent heuristic for a starting point
  new_magnitudes = (magnitudes / contrasts) * minimum_contrast
  converge_steps = new_magnitudes.copy()
  indices_needing_more_contrast = np.arange(foreground_colors.shape[0])[contrasts < minimum_contrast]
  if indices_needing_more_contrast.size < 1:
    max_iterations = 0
  new_contrasts = contrasts.copy()[indices_needing_more_contrast]
  higher_contrast_colors = foreground_colors.copy()[indices_needing_more_contrast]

  if verbose:
    info(f"\nIncreasing foreground color contrasts to {minimum_contrast}")

  for i in range(max_iterations):
    if verbose:
      colorized_contrasts = [
        ANSI_colorize(f'{contrast:0.2f}', fg=color, bg=background_color)
        for contrast, color
        in zip(new_contrasts, validate_rgb_palette(higher_contrast_colors))
      ]
      colorized_contrast_string = ' '.join(colorized_contrasts)
      info(f'{i}: {colorized_contrast_string}')

    if indices_needing_more_contrast.size < 1:  break

    _gradients = gradients[indices_needing_more_contrast]
    _new_magnitudes = new_magnitudes[indices_needing_more_contrast]

    higher_contrast_colors = (_gradients * _new_magnitudes[:,np.newaxis]) + background_color

    new_contrasts = np.apply_along_axis(_contrast, axis=1, arr=higher_contrast_colors)
    undershot_filter = new_contrasts < minimum_contrast
    indices_undershot = indices_needing_more_contrast[undershot_filter]
    indices_overshot = indices_needing_more_contrast[~undershot_filter]

    converge_steps /= 2
    new_magnitudes[indices_undershot] += converge_steps[indices_undershot]
    new_magnitudes[indices_overshot] -= converge_steps[indices_overshot]

    foreground_colors[indices_needing_more_contrast] = higher_contrast_colors

    contrast_unsatisfied_filter = np.abs(new_contrasts - minimum_contrast) > minimum_error
    indices_needing_more_contrast = indices_needing_more_contrast[contrast_unsatisfied_filter]

  return validate_rgb_palette(foreground_colors)



class TerminalImagePreview(contextlib.AbstractContextManager):
  # Print image preview to terminal using w3m
  # https://blog.z3bra.org/2014/01/images-in-terminal.html

  WIDTH_SCALAR  = 8
  HEIGHT_SCALAR = 18

  W3M_IMGDISPLAY_BIN = "/usr/lib/w3m/w3mimgdisplay"

  # If stderr is redirected, don't display a terminal image preview
  def __new__(_class, *args, **kwargs):
    if os.isatty(sys.stderr.fileno()):
      return super(TerminalImagePreview, _class).__new__(_class)
    else:
      return None


  def __init__(self, image, padding=(0,0,0,0)):
    self.image = image
    super().__init__()
    self.fd = sys.stdin.fileno()
    self.stty = termios.tcgetattr(self.fd)

    # self.border_size = border_size
    padding_top, padding_right, padding_bottom, padding_left = padding
    padding_horizontal = padding_left + padding_right
    padding_vertical = padding_top + padding_bottom
    self.offset = ( padding_left*self.WIDTH_SCALAR, padding_top*self.HEIGHT_SCALAR )

    self.term_width, self.term_height = get_terminal_size()
    self.term_width -= padding_horizontal
    self.term_height -= padding_vertical
    self.pixel_width  = self.term_width  * self.WIDTH_SCALAR
    self.pixel_height = self.term_height * self.HEIGHT_SCALAR
    image_width, image_height = self.image.size

    if image_width >= image_height:
      resize_width  = self.pixel_width
      resize_height = image_height * self.pixel_width / image_width
    else:
      resize_width  = image_width * self.pixel_height / image_height
      resize_height = self.pixel_height

    self.resize_width, self.resize_height = int(resize_width), int(resize_height)


  def display_image(self):
    with tempfile.NamedTemporaryFile(suffix=f'.jpg') as tempf:
      self.image.save(tempf.name)
      w3m_input = (
        f"0;1;{self.offset[0]};{self.offset[1]};{self.resize_width};{self.resize_height};;;;;{tempf.name}\n4;\n3;\n"
      )
      popen(self.W3M_IMGDISPLAY_BIN, stdin=w3m_input)
      popen(self.W3M_IMGDISPLAY_BIN, stdin=w3m_input) # second time to improve latching on a double-buffer terminal


  def __enter__(self):
    info(
      "\033[?1049h" # switch to secondary buffer
      "\033[?25l"   # hide cursor flashing
      "\033[0H"     # move cursor to top left
      "\033[2J"     # clear entire screen
    )

    # https://blog.nelhage.com/2009/12/a-brief-introduction-to-termios-termios3-and-stty/
    # ICANON = Canonical Mode, i.e. disabling is enabling character break "cbreak"
    # ECHO = echoing characters, disable to not print input keys
    new = termios.tcgetattr(self.fd)
    new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(self.fd, termios.TCSADRAIN, new) # Change attributes once output queue is "drained"

    return self


  def __exit__(self, exc_type, exc_value, exc_traceback):
    info(
      "\033[?25h"   # show cursor flashing
      "\033[2J"     # clear entire screen
      "\033[?1049l" # switch back to primary buffer
    )

    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.stty)
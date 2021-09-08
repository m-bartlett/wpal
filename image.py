import numpy as np
import contextlib
from util import *
from PIL import Image, ImageFilter
from colorsys import hsv_to_rgb, rgb_to_hsv
import tempfile
import termios


np.set_printoptions(precision=3, suppress=True)

BLACK=np.array([0,0,0])
WHITE=np.array([255,255,255])

ANSI = np.uint8(
  [
    list(map(int, c.partition('(')[2][:-1].split(',')))  # this is to trick my editor to show the colors
    for c in [

      "rgb(15,15,15)",
      "rgb(220,75,95)",
      # "rgb(75,220,75)",
      "rgb(00,180,80)",
      # "rgb(220,180,75)",
      # "rgb(255,120,0)",
      "rgb(237,150,37)",
      "rgb(75,100,220)",
      "rgb(120,75,220)",
      "rgb(75,170,220)",
      "rgb(220,220,220)"

      # # Muted
      # "rgb(40,40,40)",
      # "rgb(220,75,75)",
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


def ansi_colorize(message, fg='', bg=''):
  if isinstance(fg, np.ndarray):
    fg = fg.astype(np.uint8)
  fg = "38;2;{0};{1};{2}".format(*fg)
  if isinstance(bg, np.ndarray):
    bg = bg.astype(np.uint8)
  bg = "48;2;{0};{1};{2}".format(*bg)
  return f"\x1b[{bg};{fg}m{message}\x1b[0m"


def rgb2ansi_colorized_hex(rgb):
  return ansi_colorize(
    rgb2hex(rgb),
    fg=most_visible_foreground_color(rgb),
    bg=rgb
  )


def palette_as_colorized_hexcodes(palette, separator=""):
  palette = validate_rgb_palette(palette)
  return separator.join([rgb2ansi_colorized_hex(rgb) for rgb in palette])


def palette_as_foreground_on_background_ANSI_colors( foreground_colors,
                                                     background_color,
                                                     separator=""      ):
  foreground_colors = validate_rgb_palette(foreground_colors)
  background_color = validate_rgb_palette(background_color)
  return separator.join([
    ansi_colorize(rgb2hex(rgb), fg=rgb, bg=background_color)
    for rgb in foreground_colors
  ])


def palette_as_filled_blocks(palette, block_content=" ", separator=""):
  palette = validate_rgb_palette(palette)
  return separator.join([
    ansi_colorize(block_content, fg=rgb, bg=rgb)
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
    debug(
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
  width, _ = get_terminal_size()
  offset_width = (width - palette_info_width) // 2
  offset = " " * offset_width

  debug(
    offset +
    ansi_colorize(" " + spacer + rgb2hex(bg) + spacer + " ", fg=highlight, bg=bg) + " " +
    ansi_colorize(" " + spacer + rgb2hex(fg) + spacer + " ", fg=lowlight, bg=fg)
  )
  debug()

  colors = base_colors[1:-1].copy()
  colors[[1,2,3,4,5]] = colors[[2,1,5,3,4]]
  debug(offset + palette_as_foreground_on_background_ANSI_colors(colors, bg, separator=" "))
  debug(offset + palette_as_filled_blocks(colors, block_content=spacer, separator=" "))

  colors = bold_colors[1:-1].copy()
  colors[[1,2,3,4,5]] = colors[[2,1,5,3,4]]
  debug(offset + palette_as_filled_blocks(colors, block_content=spacer, separator=" "))
  debug(offset + palette_as_foreground_on_background_ANSI_colors(colors, bg, separator=" "))


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


def create_gradated_palettes(hsv_palette, value_scalars, saturation_scalars):
  # Create an array that is several duplicates of the original palette with different values
  palettes = np.tile(hsv_palette, [len(value_scalars)+1, 1, 1])

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
    debug(f"\nIncreasing contrast to {minimum_contrast}")

  for i in range(max_iterations):
    if verbose:
      debug(f'{i}: {new_contrasts}->{minimum_contrast} ', end='')
      debug(
        palette_as_foreground_on_background_ANSI_colors(
          foreground_colors=higher_contrast_colors,
          background_color=background_color,
          separator=" "
        )
      )

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

  WIDTH_SCALAR  = 7
  HEIGHT_SCALAR = 18

  W3M_IMGDISPLAY_BIN = "/usr/lib/w3m/w3mimgdisplay"

  # If stderr is redirected, don't display a terminal image preview
  def __new__(_class, *args, **kwargs):
    if os.isatty(sys.stderr.fileno()):
      return super(TerminalImagePreview, _class).__new__(_class)
    else:
      return None


  def __init__(self, image):
    self.image = image
    super().__init__()
    self.fd = sys.stdin.fileno()
    self.stty = termios.tcgetattr(self.fd)
    self.TERM_WIDTH, self.TERM_HEIGHT = get_terminal_size()
    self.WIDTH, self.HEIGHT = self.TERM_WIDTH*self.WIDTH_SCALAR , self.TERM_HEIGHT*self.HEIGHT_SCALAR


  def display_image(self):
    with tempfile.NamedTemporaryFile(suffix=f'.jpg') as tempf:
      self.image.save(tempf.name)
      stdin = f"0;1;0;0;{self.WIDTH};{self.HEIGHT};;;;;{tempf.name}\n4;\n3;\n"
      popen(self.W3M_IMGDISPLAY_BIN, stdin=stdin)


  def __enter__(self):
    debug(
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

    self.display_image()
    return self


  def __exit__(self, exc_type, exc_value, exc_traceback):
    debug(
      "\033[?25h"   # show cursor flashing
      "\033[2J"     # clear entire screen
      "\033[?1049l" # switch back to primary buffer
    )

    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.stty)
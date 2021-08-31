import tempfile
from util import popen

def xrdb_merge(Xresource_colors):
  with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp:
    temp.writelines(
      f"*{color_name}: {color_hex}\n"
      for color_name, color_hex in Xresource_colors.items()
    )
    temp.flush()
    return popen(f"xrdb -merge {temp.name}")
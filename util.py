import subprocess
import shlex
import os
import sys
from pathlib import Path


def printerr(*args, **kwargs):
  kwargs['file']=sys.stderr
  print(*args, **kwargs)


def get_current_wallpaper():
  with Path("~/.config/nitrogen/bg-saved.cfg").expanduser().open('r') as wp_file:
    wp_file.readline()
    return wp_file.readline().split('=')[1].strip()


def get_terminal_size():
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
    return int(cr[1]), int(cr[0])


def popen(cmdline, stdin=None, **kwargs):
  if isinstance(stdin, str): stdin=stdin.encode()
  return subprocess.run(
    shlex.split(cmdline),
    input=stdin,
    capture_output=True,
    **kwargs
  )
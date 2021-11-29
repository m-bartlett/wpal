import subprocess
import shlex
import os
import sys
from pathlib import Path

EXECUTABLE_DIRECTORY = Path(__file__).resolve(strict=True).parent
EXECUTABLE_NAME = EXECUTABLE_DIRECTORY.name

def info(*args, **kwargs):
    kwargs['file']=sys.stderr
    print(*args, **kwargs)


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
    list(map(os.path.expanduser, shlex.split(cmdline))),
    input=stdin,
    capture_output=True,
    **kwargs
  )
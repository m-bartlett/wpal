import subprocess
import shlex
import os
import sys
import fcntl, termios, struct, shutil  # for terminal size
from pathlib import Path

EXECUTABLE_DIRECTORY = Path(__file__).resolve(strict=True).parent
EXECUTABLE_NAME = EXECUTABLE_DIRECTORY.name

def info(*args, **kwargs):
    kwargs['file']=sys.stderr
    print(*args, **kwargs)


def fail(s, **kwargs):
    info(f'\033[31m{s}\033[0m', **kwargs)
    sys.exit(1)


def get_terminal_size():
    for fd in 0,1,2:
        try: terminal_size = struct.unpack( 'hh',
                                            fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234') )
        except OSError: continue
    else:
        try:    terminal_size = shutil.get_terminal_size()
        except: terminal_size = (os.getenv('LINES', 25), os.getenv('COLUMNS', 80))
    return tuple(map(int, terminal_size))


def popen(cmdline, stdin=None, **kwargs):
    if isinstance(stdin, str):
        stdin=stdin.encode()
    return subprocess.run( list(map(os.path.expanduser, shlex.split(cmdline))),
                           input=stdin,
                           capture_output=True,
                           **kwargs )

def popen_blocking(cmdline, stdin=None, **kwargs):
    exit_code = subprocess.call( list(map(os.path.expanduser, shlex.split(cmdline))),
                                 stdin=stdin,
                                 # shell=True,
                                 text='utf-8',
                                 **kwargs )
    if exit_code != 0:
        info(f"WARNING: '{cmdline}' returned non-zero exit code")
    return exit_code
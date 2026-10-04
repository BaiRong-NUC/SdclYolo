import _bootstrap
from sdcl.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *sys.argv[1:]]))


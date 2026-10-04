import _bootstrap
from sdcl.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(["audit", *sys.argv[1:]]))


import _bootstrap
from sdcl.cli import main
import sys

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))


"""Convenience entrypoint for `python -m simpledet`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

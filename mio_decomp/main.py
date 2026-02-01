from .src.app import app


def main():  # This method exists so I can import it in ./__main__.py, so I can support `python -m` while still only having to change this in one place.
    app(name="mio-decomp")


if __name__ == "__main__":
    main()

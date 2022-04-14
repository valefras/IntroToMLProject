import argparse


def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m competition",
        description="Competition",
    )
    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    r"""Main function"""


if __name__ == "__main__":
    main(get_args())

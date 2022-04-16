import argparse
from competition.solution_b import solution_b
from competition.solution_a import solution_a

def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m competition",
        description="Competition",
    )

    #subparsers

    subparsers = parser.add_subparsers(help="sub-commands help")
    solution_b.configure_subparsers(subparsers)
    solution_a.configure_subparsers(subparsers)

    #parsing arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    r"""Main function"""
    args.func(args,)

if __name__ == "__main__":
    main(get_args())

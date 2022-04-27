import argparse
from competition.AE import AE
from competition.classifier import classifier
from competition.comp_classifier import comp_classifier

def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m competition",
        description="Competition",
    )

    #subparsers

    subparsers = parser.add_subparsers(help="sub-commands help")
    AE.configure_subparsers(subparsers)
    classifier.configure_subparsers(subparsers)
    comp_classifier.configure_subparsers(subparsers)

    #parsing arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    r"""Main function"""
    args.func(args,)

if __name__ == "__main__":
    main(get_args())

import argparse
from competition.AE import AE
from competition.classifier import classifier
from competition.comp_classifier import comp_classifier
from competition.comp_AE import comp_AE
from competition.cAE_revised import cAE_revised
from competition.comp_AE128 import comp_AE128
from competition.comp_classifier_2 import comp_classifier_2
from competition.resnet18 import resnet18
from competition.resnet34 import resnet34
from competition.resnet50 import resnet50
from competition.resnet101 import resnet101
from competition.resnet152 import resnet152
from competition.convnext_tiny import convnext_tiny
from competition.convnext_small import convnext_small
from competition.efficientnet_b1 import efficientnet_b1
from competition.efficientnet_b2 import efficientnet_b2


def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m competition",
        description="Competition",
    )

    # subparsers

    subparsers = parser.add_subparsers(help="sub-commands help")
    AE.configure_subparsers(subparsers)
    classifier.configure_subparsers(subparsers)
    comp_classifier.configure_subparsers(subparsers)
    comp_AE.configure_subparsers(subparsers)
    comp_AE128.configure_subparsers(subparsers)
    cAE_revised.configure_subparsers(subparsers)
    comp_classifier_2.configure_subparsers(subparsers)
    convnext_tiny.configure_subparsers(subparsers)
    convnext_small.configure_subparsers(subparsers)
    resnet18.configure_subparsers(subparsers)
    resnet34.configure_subparsers(subparsers)
    resnet50.configure_subparsers(subparsers)
    resnet101.configure_subparsers(subparsers)
    resnet152.configure_subparsers(subparsers)
    efficientnet_b1.configure_subparsers(subparsers)
    efficientnet_b2.configure_subparsers(subparsers)

    # parsing arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    r"""Main function"""
    args.func(args,)


if __name__ == "__main__":
    main(get_args())

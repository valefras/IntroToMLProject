

def configure_subparsers(subparsers):
  r"""Configure a new subparser for our second solution of the competition.
  Args:
    subparsers: subparser
  """

  """
  Subparser parameters:
  Args:
  """
  parser = subparsers.add_parser("solution_b", help="Train the Solution B model")
  parser.add_argument(
      "--test", action='store_true', default=False, help="Test the model"
    )


  parser.set_defaults(func=main)






def main(args):
    print("Main function of Solution B")

    print("Parameters given:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))

    print(args.test)

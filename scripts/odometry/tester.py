import __init_path__

from scripts.base_tester import BaseTester


if __name__ == '__main__':
    parser = BaseTester.get_parser()
    args, other_args = parser.parse_known_args()

    tester = BaseTester(**vars(args))
    tester.test()

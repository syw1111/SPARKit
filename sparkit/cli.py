import argparse

from sparkit.sphase_order.pipeline import run_sphase_order



def main():

    parser=argparse.ArgumentParser(
        prog="SPARKit"
    )


    sub=parser.add_subparsers(
        dest="command"
    )


    sphase=sub.add_parser(
        "sphase_order"
    )


    sphase.add_argument(
        "--bam",
        required=True
    )


    sphase.add_argument(
        "--prefix",
        required=True
    )


    sphase.add_argument(
        "--genome",
        default="hg38"
    )


    args=parser.parse_args()


    if args.command=="sphase_order":

        run_sphase_order(
            args.bam,
            args.prefix,
            args.genome
        )
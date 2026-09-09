#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import pysam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract reads from a BAM file for which the CB tag value exists in the provided barcode list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True,
                       help="Input BAM")
    parser.add_argument("-o", "--out-prefix", required=True )
    parser.add_argument("-b", "--barcodes", default="NEG.txt")
    parser.add_argument("-s", "--suffix", default=".CB1K.bam")
    parser.add_argument("-t", "--tag", default="CB")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("-@", "--threads", type=int, default=16)
    return parser.parse_args()


def load_barcodes(path):
    barcodes = set()
    with open(path, "r") as f:
        for line in f:
            bc = line.strip()
            if bc and not bc.startswith("#"):
                barcodes.add(bc)
    return barcodes


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit("[ERROR] Input BAM file does not exist: %s" % args.input)
    if not os.path.isfile(args.barcodes):
        sys.exit("[ERROR] Input barcode file does not exist: %s" % args.barcodes)

    barcodes = load_barcodes(args.barcodes)
    if not barcodes:
        sys.exit("[ERROR] The barcode file is empty: %s" % args.barcodes)
    sys.stderr.write("[INFO] Loaded %d barcodes\n" % len(barcodes))

    out_file = args.out_prefix + args.suffix
    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    total = kept = 0
    read_mode = "rb"
    write_mode = "wb"

    with pysam.AlignmentFile(args.input, read_mode, threads=args.threads) as in_bam:
        with pysam.AlignmentFile(out_file, write_mode, template=in_bam,
                                 threads=args.threads) as out_bam:
            for read in in_bam.fetch(until_eof=True):
                total += 1
                if read.has_tag(args.tag) and read.get_tag(args.tag) in barcodes:
                    out_bam.write(read)
                    kept += 1

if __name__ == "__main__":
    main()
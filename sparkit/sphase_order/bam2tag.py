#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import re
import argparse
import subprocess

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')
REF_CONSUMING = frozenset('MDN=X')          


def parse_cigar(cigar_str):
    if cigar_str == '*':
        return 0
    total = 0
    for length_str, op in CIGAR_RE.findall(cigar_str):
        if op in REF_CONSUMING:
            total += int(length_str)
    return total


def get_args():
    p = argparse.ArgumentParser(
        description='Output BED intervals and CB statistics per read from samtools view SAM stream.')
    p.add_argument('-b', '--bam',required=True)
    p.add_argument('-p', '--prefix', required=True)
    p.add_argument('-t', '--barcode-tag', default='CB')
    p.add_argument('-q', '--min-mapq', type=int, default=0)
    p.add_argument('--no-header', action='store_true')
    return p.parse_args()


def main():

    args = get_args()

    frag_file = f'{args.prefix}.fragments.txt'
    cnt_file = f'{args.prefix}.CB_counts.txt'

    tag_prefix = f'{args.barcode_tag}:Z:'

    cb_counts = {}
    n_total = n_written = 0


    samtools_cmd = [
        "samtools",
        "view",
        args.bam
    ]


    process = subprocess.Popen(samtools_cmd,stdout=subprocess.PIPE,text=True)


    with open(frag_file, 'w') as out_handle:

        if not args.no_header:
            out_handle.write(
                'chr\tstart\tend\tCB\tcount\n'
            )


        for line in process.stdout:

            if line.startswith('@'):
                continue


            n_total += 1

            fields = line.rstrip('\n').split('\t')


            if len(fields) < 12:
                continue


            chrom, pos, mapq, cigar = (
                fields[2],
                fields[3],
                fields[4],
                fields[5]
            )


            if chrom == '*' or pos == '0' or cigar == '*':
                continue


            if int(mapq) < args.min_mapq:
                continue


            pos = int(pos)

            aln_len = parse_cigar(cigar)


            if aln_len <= 0:
                continue


            end_pos = pos + aln_len - 1


            cb_tag = None


            for opt in fields[11:]:

                if opt.startswith(tag_prefix):

                    cb_tag = opt[len(tag_prefix):]

                    break


            if cb_tag is None or cb_tag == '-':
                continue


            cb_counts[cb_tag] = (
                cb_counts.get(cb_tag, 0) + 1
            )


            out_handle.write(
                f'{chrom}\t{pos}\t{end_pos}\t{cb_tag}\t1\n'
            )


            n_written += 1


    process.stdout.close()

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "samtools view failed"
        )


    with open(cnt_file, 'w') as fh:

        fh.write('CB\tcount\n')

        for cb, cnt in sorted(
            cb_counts.items(),
            key=lambda x: -x[1]
        ):

            fh.write(
                f'{cb}\t{cnt}\n'
            )


if __name__ == '__main__':
    main()
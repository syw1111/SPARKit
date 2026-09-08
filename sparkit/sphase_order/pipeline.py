import os
import subprocess


def run_command(cmd):

    print(
        "Running:",
        " ".join(cmd)
    )

    subprocess.run(
        cmd,
        check=True
    )



def run_sphase_order(
        bam,
        prefix,
        genome
):


    gc_file=f"{genome}.gc.200kb.bedgraph"


    if not os.path.exists(gc_file):

        cmd=f"""
        bedtools makewindows \
        -g {genome}.chrom.sizes.txt \
        -w 200000 |
        grep -P '^chr([0-9]+)\\t' |
        bedtools nuc \
        -fi {genome}.fa \
        -bed - |
        awk 'NR>1{{print $1"\\t"$2"\\t"$3"\\t"$5}}' \
        > {gc_file}
        """

        subprocess.run(
            cmd,
            shell=True,
            check=True
        )


    run_command(
        [
        "SPARKit-bam2tag",
        "--bam",
        bam,
        "--prefix",
        prefix
        ]
    )


    run_command(
        [
        "SPARKit-Sphase-order",
        "--fragments",
        prefix+".fragments.txt",

        "--chrom-sizes",
        genome+".chrom.sizes.txt",

        "--anchor-bedgraph",
        gc_file,

        "--outdir",
        prefix
        ]
    )
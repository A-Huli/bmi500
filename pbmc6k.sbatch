#!/bin/bash
#SBATCH -J pbmc6k
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o pbmc6k-%j.out

source bmi500/bin/activate

/usr/bin/time -v python scanpy_pbmc.py --data-dir data --data-set pbmc6k --out-dir data --num-threads 1
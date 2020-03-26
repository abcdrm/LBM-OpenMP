# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=28
module load languages/intel/2018-u3
module load GCC/7.2.0-2.29
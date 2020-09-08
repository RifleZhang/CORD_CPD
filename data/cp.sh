type=$1
folder=cp_small_${type}
python generate_cp.py --num-train 500 --num-valid 100 --num-test 100 --out ${folder} --change-type ${type}

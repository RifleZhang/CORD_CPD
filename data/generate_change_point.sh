python generate_cp.py --num-train 500 --num-valid 100 --num-test 100 --out cp_loc --change-type loc
python generate_cp.py --num-train 500 --num-valid 100 --num-test 100 --out cp_vel --change-type vel
python generate_cp.py --num-train 500 --num-valid 100 --num-test 100 --out cp_edge --change-type edge
# combine generated change-point time series to form a dataset
python generate_dataset.py

import os
list_benchmarks = [
    "iris_3x2",
    "iris_4x2",
    "iris_10x2",
    "iris_15x2",
    "seeds_2x1",
    "seeds_4x1",
    "seeds_5x1",
    "seeds_6x1",
    "seeds_10x1",
    "seeds_15x1"
]

# run command

for benchmark in list_benchmarks:
    for solver in ["milp", "esbmc"]:
        for eps in [0.01, 0.03, 0.05]:
            print(f"Running benchmark: {benchmark} with solver {solver}")
            response = os.system(f"python3 Quadapter_robustness_main.py --dataset {benchmark} --arch 1blk_10 --sample_id 25 --eps {eps} --preimg_mode milp --verify_mode {solver} --ifRelax 0 --outputPath ./output/")
            if response != 0:
                print(f"Error running benchmark: {benchmark} with solver {solver}")
            else:
                print(f"Successfully ran benchmark: {benchmark} with solver {solver}")

        

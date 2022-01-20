from glob import glob
import os
import shutil

def data_move(current_dir, target_dir):
    shutil.move(current_dir, target_dir)


def main(dtype):
    current_dir = r"D:/IR_data/1207_hyundai_ir_data/sequential_4/*.jpg".format(dtype)
    target_dir = r"D:/IR_data/1207_hyundai_ir_data/raw_data/8bit"
    data = glob(current_dir)
    for i in range(len(data)):
        data_move(data[i], target_dir)

if __name__ == "__main__":
    main("RGB")
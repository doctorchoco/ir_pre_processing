import json
import os



def main():
    dict = {}
    false_class_dir = r"D:/IR_data/1207_hyundai_ir_data/labeled/8bit/Pos/"
    true_class_dir = r"D:/IR_data/1207_hyundai_ir_data/labeled/8bit/Neg/"
    false_data_dir = os.listdir(false_class_dir)
    true_data_dir = os.listdir(true_class_dir)

    dict["0"] = false_data_dir
    dict["1"] = true_data_dir

    save_dir = r"D:/IR_data/1207_hyundai_ir_data/labeled/8bit/label.json"
    with open(save_dir, "w") as json_file:
        json.dumps(dict, json_file)


if __name__ == "__main__":
    main()
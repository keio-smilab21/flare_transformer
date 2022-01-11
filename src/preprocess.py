""" Preprocess physical features and magnetogram images for model input"""
import argparse
import json
import datetime
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


def preprocess_magnetogram(img_path):
    """tmp"""
    img = Image.open(img_path)
    transform = transforms.Compose(
        [
            transforms.Resize(512)
        ]
    )
    img = transform(img)
    img = np.array(img)
    img = img[:, :, 0]
    img = img[np.newaxis, :, :]
    return img


def get_time(str_time):
    """tmp"""
    return datetime.datetime.strptime(str_time, '%d-%b-%Y %H')


def sub_1h(str_time):
    """tmp"""
    time = get_time(str_time)
    next_time = time - datetime.timedelta(hours=1)
    next_time = datetime.datetime.strftime(next_time, '%d-%b-%Y %H')
    return next_time


def get_fancy_index(num, db, current_data):
    """tmp"""
    index = list(range(max(num-horizon+1, 0), num))
    index.append(num)
    candidate_time = [data["time"] for data in db[max(num-horizon+1, 0):num]]
    candidate_time.append(current_data["time"])

    fancy_index = []
    time = current_data["time"]
    for i in range(horizon):
        if time in candidate_time:
            fancy_index.append(index[candidate_time.index(time)])
        else:
            fancy_index.append(fancy_index[i-1])
        time = sub_1h(time)

    return fancy_index


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path',
                        default='data/ft_database_all17.jsonl')
    parser.add_argument('--path', default='data/data_')
    parser.add_argument('--start_year', default='2010')
    parser.add_argument('--end_year', default='2017')
    parser.add_argument('--horizon', default='48')
    args = parser.parse_args()
    database_path = args.database_path
    path = args.path
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    horizon = int(args.horizon)

    # find index
    year = [str(y) for y in range(start_year, end_year+1)]
    idx = {str(start_year-1): {"start": 0, "end": 0}}
    for y in year:
        idx[y] = {"start": 0, "end": 0}

    year_tmp = year.copy()
    with open(database_path) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            year_str = data["time"][-7:-3]
            if year_str in year_tmp:
                idx[year_str]["start"] = i
                idx[str(int(year_str)-1)]["end"] = i - 1
                year_tmp.remove(year_str)
    idx[str(end_year)]["end"] = i  # last data

    del idx[str(start_year-1)]

    with open(database_path) as f:
        db = f.readlines()
        db = list(map(json.loads, db))

    # Magnetogram Image
    for y in year:
        db_split = db[idx[y]["start"]:idx[y]["end"]+1]
        image_data = []
        output_path = path + y + "_magnetogram.npy"
        print(output_path)
        for i, data in enumerate(tqdm(db_split, total=len(db_split))):
            # image
            image_data.append(preprocess_magnetogram(data["magnetogram"]))
        np.save(output_path, image_data)

    # Feature and label
    for y in year:
        db_split = db[idx[y]["start"]:idx[y]["end"]+1]
        image_data = []
        output_feat_path = path + y + "_feat.csv"
        output_label_path = path + y + "_label.csv"
        with open(output_feat_path, "w") as wwf:
            with open(output_label_path, "w") as wf:
                for i, data in enumerate(tqdm(db_split, total=len(db_split))):
                    # feature
                    wwf.write(data["feature"])
                    wwf.write("\n")
                    # label
                    wf.write(data["flag"])
                    wf.write("\n")

    # Window index
    fancy_index = []
    for y in year:
        db_split = db[idx[y]["start"]:idx[y]["end"]+1]
        output_window_path = path + y + "_window_" + str(horizon) + ".csv"
        with open(output_window_path, "w") as wf:
            for i, data in enumerate(tqdm(db_split)):
                fancy_index = np.array(get_fancy_index(
                    i+idx[y]["start"], db, data)) - idx[y]["start"]
                fancy_index = [max(x, 0) for x in fancy_index]
                wf.write(",".join(map(str, fancy_index)))
                wf.write("\n")

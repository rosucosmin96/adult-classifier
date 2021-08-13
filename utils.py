import os
import csv
import numpy as np
import random
import math

from PIL import Image, ImageOps


def get_filelist(root_path, ext):
    if type(ext) is str:
        ext = [ext]

    for i in range(len(ext)):
        if len(ext[i]) > 0 and ext[i][0] != '.':
            ext[i] = '.' + ext[i]

        ext[i] = ext[i].lower()

    result = []

    for (dirpath, dirname, filenames) in os.walk(root_path):
        for filename in filenames:
            fullpath = os.path.join(dirpath, filename)

            for extension in ext:
                if filename.endswith(extension) and filename.startswith('._') is False:
                    result.append(fullpath)
                    break

    return result


def generate_annotation(root_dir, annotation_file='adult_annotation.csv'):
    assert os.path.isdir(root_dir)
    anno_path = os.path.join(root_dir, annotation_file)

    filelist = get_filelist(root_dir, '.jpg')
    filelist.extend(get_filelist(root_dir, '.png'))

    with open(anno_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['FilePath', 'Adult'])

        for file in filelist:
            file_data = file.split('\\')[-1]
            label = 1 if int(file_data.split('_')[0]) >= 18 else 0

            abspath = os.path.abspath(file)

            writer.writerow([abspath, label])


def split_annotation(annotation_file, train_prec=0.7, val_prec=0.2):
    assert os.path.exists(annotation_file)

    root_path = '/'.join(annotation_file.split('/')[:-1])
    train_path = os.path.join(root_path, 'train_annotation.csv')
    val_path = os.path.join(root_path, 'val_annotation.csv')
    test_path = os.path.join(root_path, 'test_annotation.csv')

    file_list = []
    with open(annotation_file) as orig_csv:
        csv_reader = csv.reader(orig_csv, delimiter=';')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count = 1
            else:
                file_list.append((row[0], row[1]))

    random.shuffle(file_list)
    orig_len = len(file_list)

    train_len = int(train_prec * orig_len)
    train_list = file_list[:train_len + 1]
    writeListToCSV(train_list, train_path)

    val_len = int(val_prec * orig_len)
    val_list = file_list[(train_len + 1):(train_len + val_len + 1)]
    writeListToCSV(val_list, val_path)

    test_list = file_list[(train_len + val_len + 1):]
    writeListToCSV(test_list, test_path)


def writeListToCSV(file_list, file_path):
    with open(file_path, mode='w', newline='') as train_csv:
        writer = csv.writer(train_csv, delimiter=';')
        writer.writerow(['FilePath', 'Adult'])

        for row in file_list:
            writer.writerow([row[0], row[1]])


def processImg(img_orig, w=224, h=224, resize=False):
    img = img_orig.convert('RGB')

    if resize:
        base_width, base_height = w, h
        img_width, img_height = img.size[0], img.size[1]

        if img_width >= img_height:
            width = base_width
            width_percent = float(width) / img_width
            height = int(img_height * width_percent)
        else:
            height = base_height
            height_percent = float(height) / img_height
            width = int(img_width * height_percent)

        if width <= 0:
            width = 1
        elif height <= 0:
            height = 1

        img = img.resize((width, height), Image.BICUBIC)

        delta_width = base_width - width
        delta_height = base_height - height
        half_delta_w = delta_width // 2
        hald_delta_h = delta_height // 2

        padding = (half_delta_w, hald_delta_h, delta_width - half_delta_w, delta_height - hald_delta_h)
        img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    img = ImageOps.grayscale(img)

    return img


def normalizeImg(img):
    np_img = np.array(img)

    # [width, height] -> [1, 1, width, height]
    np_img = np.expand_dims(np.expand_dims(np_img, axis=0), axis=0)

    return np_img / 255


def compute_class_weights(labels_dict, mu=.6):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


if __name__ == '__main__':
    root_dir = r'./data/UTKFace'
    generate_annotation(root_dir)
    split_annotation(r'./data/UTKFace/adult_annotation.csv')

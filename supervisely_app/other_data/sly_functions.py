from detectron2.structures import BoxMode

import json
from PIL import Image

import supervisely_lib as sly


def get_image_info(image_path):
    im = Image.open(image_path)
    width, height = im.size

    return width, height


def get_objects_on_image(ann, all_classes):
    objects_on_image = []

    for label in ann.labels:
        rect = label.geometry.to_bbox()
        curr_poly = label.geometry.convert(sly.Polygon)[0].exterior_np.tolist()
        new_poly = [point[::-1] for point in curr_poly]

        obj = {
            "bbox": [rect.left, rect.top, rect.right, rect.bottom],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [new_poly],
            "category_id": all_classes[label.obj_class.name],
        }

        objects_on_image.append(obj)

    return objects_on_image


def get_items_by_set_path(set_path):
    files_by_datasets = {}
    with open(set_path, 'r') as train_set_file:
        set_list = json.load(train_set_file)

        for row in set_list:
            existing_items = files_by_datasets.get(row['dataset_name'], [])
            existing_items.append(row['item_name'])
            files_by_datasets[row['dataset_name']] = existing_items

    return files_by_datasets


def convertStringToNumber(s):
    return int.from_bytes(s.encode(), 'little')


def convert_data_to_detectron(project_seg_dir_path, set_path):
    dataset_dicts = []

    project = sly.Project(directory=project_seg_dir_path, mode=sly.OpenMode.READ)
    project_meta = project.meta

    all_classes = {}

    for class_index, obj_class in enumerate(project_meta.obj_classes):
        all_classes[obj_class.name] = class_index

    files_by_datasets = get_items_by_set_path(set_path=set_path)

    datasets_list = project.datasets
    for current_dataset in datasets_list:
        current_dataset_name = current_dataset.name

        items_in_dataset = files_by_datasets.get(current_dataset_name, [])

        for current_item in items_in_dataset:
            record = {
                "file_name": current_dataset.get_item_path(current_item),
                "image_id": convertStringToNumber(f"{set_path}_{current_dataset.get_item_path(current_item)}")
            }

            width, height = get_image_info(record["file_name"])
            record["height"] = width
            record["width"] = height

            ann_path = current_dataset.get_ann_path(current_item)
            ann = sly.Annotation.load_json_file(ann_path, project_meta)

            record["annotations"] = get_objects_on_image(ann, all_classes)
            dataset_dicts.append(record)

    return dataset_dicts


print(convert_data_to_detectron('/app_debug_data/data/Lemons (Annotated)_seg',
                                '/app_debug_data/data/artifacts/info/train_set.json'))

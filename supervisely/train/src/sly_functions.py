import functools

import numpy as np
import pycocotools.mask

from detectron2.structures import BoxMode

import supervisely_lib as sly


def mask_to_image_size(label, existence_mask, img_size):
    mask_in_images_coordinates = np.zeros(img_size, dtype=bool)  # size is (h, w)

    row, column = label.geometry.origin.row, label.geometry.origin.col  # move mask to image space
    mask_in_images_coordinates[row: row + existence_mask.shape[0], column: column + existence_mask.shape[1]] = \
        existence_mask

    return mask_in_images_coordinates


def get_objects_on_image(ann, all_classes):
    objects_on_image = []

    for label in ann.labels:
        rect = label.geometry.to_bbox()
        # curr_poly = np.asarray(label.geometry.convert(sly.Polygon)[0].exterior_np.tolist())
        # new_poly = np.asarray([point[::-1] for point in curr_poly])
        # new_poly = new_poly.ravel().tolist()

        seg_mask = np.asarray(label.geometry.convert(sly.Bitmap)[0].data)
        seg_mask_in_image_coords = np.asarray(mask_to_image_size(label, seg_mask, ann.img_size))

        rle_seg_mask = pycocotools.mask.encode(np.asarray(seg_mask_in_image_coords, order="F"))
        # seg_mask = binary_mask_to_rle(seg_mask)
        # seg_mask['counts'] = counts

        obj = {
            "bbox": [rect.left, rect.top, rect.right, rect.bottom],
            "bbox_mode": BoxMode.XYXY_ABS,
            # "segmentation": [new_poly],
            "segmentation": rle_seg_mask,
            "category_id": all_classes[label.obj_class.name],
        }

        objects_on_image.append(obj)

    return objects_on_image


# def get_model_config_path(state):
#     models_by_dataset = step05_models.get_pretrained_models()[state["pretrainedDataset"]]
#     selected_model = next(item for item in models_by_dataset
#                           if item["model"] == state["selectedModel"][state["pretrainedDataset"]])
#
#     if selected_model.get('config').endswith('.py'):
#         return selected_model.get('config')
#
#     if state["pretrainedDataset"] == 'COCO':
#         par_folder = 'COCO-InstanceSegmentation'
#         return os.path.join(par_folder, selected_model.get('config'))
#
#     elif state["pretrainedDataset"] == 'LVIS':
#         par_folder = 'LVISv0.5-InstanceSegmentation'
#         return os.path.join(par_folder, selected_model.get('config'))
#
#     elif state["pretrainedDataset"] == 'Cityscapes':
#         par_folder = 'Cityscapes'
#         return os.path.join(par_folder, selected_model.get('config'))
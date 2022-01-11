# import supervisely_lib as sly
#
# my_app = sly.AppService()
# api = my_app.public_api
# task_id = my_app.task_id
#
# test_project_dir = './test_project_dir'
# test_project_dir_seg = './test_project_dir_seg'
#
#
# sly.fs.clean_dir(test_project_dir)
# sly.fs.clean_dir(test_project_dir_seg)
#
# sly.download_project(api, 8491, test_project_dir,
#                      cache=my_app.cache, save_image_info=True)
#
# sly.Project.to_segmentation_task(
#             test_project_dir, test_project_dir_seg,
#             target_classes=['lemon'],
#
#         )
#
# sly.Project.to_segmentation_task(test_project_dir)


import numpy as np
from itertools import groupby


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


test_list_1 = np.array([
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 1]]
)
test_list_2 = np.array([1, 1, 1, 1, 1, 1, 0])

print(binary_mask_to_rle(test_list_1))
print(binary_mask_to_rle(test_list_2))
#

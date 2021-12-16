import json
import supervisely_lib as sly


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 12274

    # get model info
    response = api.task.send_request(task_id, "get_info", data={}, timeout=60)
    print("APP returns data:")
    print(response)

    # get masks for image by url
    response = api.task.send_request(task_id, "inference_image_url", data={
        'image_url': 'https://img.icons8.com/color/1000/000000/deciduous-tree.png'
    }, timeout=60)
    print("APP returns data:")
    print(response)


if __name__ == "__main__":
    main()

import supervisely_lib as sly
import sly_globals as g
import ui

import os


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    data = {}
    state = {}
    data["taskId"] = g.task_id

    ui.init(data, state)  # init data for UI widgets
    g.my_app.compile_template(g.root_source_dir)
    g.my_app.run(data=data, state=state)


# check AP type
# test every model with py configs

# add stop train button
# add continue train button
# fix upload models progress step bag
# reset config button

if __name__ == "__main__":
    sly.main_wrapper("main", main)


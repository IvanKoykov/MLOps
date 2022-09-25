from clearml import PipelineController


def run_pipe():
    pipe = PipelineController(
      name="Training pipeline", project="segmentation_people", version="0.0.1"
    )

    pipe.add_parameter(
        "id",
        "617f60ea548943149a09b737acb9d674",
    )

    pipe.set_default_execution_queue("default")

    pipe.add_step(
        name='data_validation',
        # parents=['stage_data', ],
        base_task_project='segmentation_people',
        base_task_name='data_validation',
        parameter_override={
            'General/dataset_id': "${pipeline.id}"},
    )
    pipe.add_step(
        name='data_preparation',
        parents=['data_validation'],
        base_task_project='segmentation_people',
        base_task_name='data_preparation',
        parameter_override={
            'General/dataset_id': "${pipeline.id}"},
    )

    pipe.add_step(
        name='training_step',
        parents=['data_preparation'],
        base_task_project='segmentation_people',
        base_task_name='training',
        parameter_override={
            'General/dataset_id': "${data_preparation.parameters.General/output_dataset_id}"},  # noqa: E501
    )

    pipe.start_locally(run_pipeline_steps_locally=True)
    print("done")


if __name__ == "__main__":
    run_pipe()

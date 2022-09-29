import clearml


def push():
    dataset = clearml.Dataset.create(
        dataset_project="segmentation_people",
        dataset_name="people",
        description="dataset_for_segmentation_people",
    )
    dataset.add_files("data")
    dataset.upload(verbose=True)
    dataset.finalize()


if __name__ == "__main__":
    print("done")
    push()

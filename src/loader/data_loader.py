class DataLoader:
    def __init__(self, dataset_name, dataset_config, split="test[:100]"):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.data = None

    def load_data(self):
        """
        Load data from Hugging Face datasets.
        """
        from datasets import load_dataset

        try:
            self.data = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load dataset {self.dataset_name!r} "
                f"with config {self.dataset_config!r} and split {self.split!r}."
            ) from error

        return self.data

    def get_data(self):
        """
        Get the loaded dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data
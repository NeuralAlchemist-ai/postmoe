class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """
        Load data from Hugging Face datasets.
        """
        try:
            from datasets import load_dataset
            self.data = load_dataset(self.data_path)
            print(f"Successfully loaded dataset: {self.data_path}")
        except Exception as e:
            print(f"Error loading dataset {self.data_path}: {e}")

    def get_data(self):
        """
        Get the loaded dataset.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data
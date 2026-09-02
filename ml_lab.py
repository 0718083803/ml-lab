import pandas as pd

class DatasetLoader:
    def __init__(self ,file_path):
        self.file_path = file_path
    def load_dataset(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dataset loaded successfully")
            return self.data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def explore_dataset(self):
        print("Dataset Information: ")
        print(self.data.info())
        
        print("Dataset Description: ")
        print(self.data.describe())
        
        print("Dataset Head: ")
        print(self.data.head())
        
        print("Dataset Shape")
        print(self.data.shape)

        print("\n--- Column names ---")
        print(self.data.columns.tolist())

        print("\n--- Data types ---")
        print(self.data.dtypes)
        
        print("\n--- Missing values ---")
        print(self.data.isnull().sum())
        

loader = DatasetLoader("Teen_Mental_Health_Dataset.csv")
data = loader.load_dataset()
if data is not None:
    loader.explore_dataset()
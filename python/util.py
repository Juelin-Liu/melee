import numpy as np

def load_ground_truth(file_path):
    # Read the binary file
    with open(file_path, 'rb') as f:
        # Read the number of rows from the first 4 bytes
        num_rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_bytes = num_rows * num_cols * 4
        labels = np.frombuffer(f.read(num_bytes), dtype=np.int32)
        distances = np.frombuffer(f.read(num_bytes), dtype=np.single)
        
        # Reshape the data into rows
        labels = labels.reshape((num_rows, -1))
        distances = distances.reshape((num_rows, -1))
        
        return labels, distances

def load_data(file_path: str, max_elements: int):
    # Read the binary file
    with open(file_path, 'rb') as f:
        # Read the number of rows from the first 4 bytes
        num_rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
        num_rows = int(min(max_elements, num_rows))
        # print(f"loading {num_rows=} from {file_path}")

        if file_path.endswith("fbin"):
            num_bytes = num_rows * num_cols * 4
            queries = np.frombuffer(f.read(num_bytes), dtype=np.single)
            queries = queries.reshape((num_rows, num_cols))
            return queries
        elif file_path.endswith("u8bin"):
            num_bytes = num_rows * num_cols
            queries = np.frombuffer(f.read(num_bytes), dtype=np.uint8)
            queries = queries.reshape((num_rows, num_cols))
            return queries.astype(np.single)
        elif file_path.endswith("i8bin"):
            num_bytes = num_rows * num_cols
            queries = np.frombuffer(f.read(num_bytes), dtype=np.int8)
            queries = queries.reshape((num_rows, num_cols))
            return queries.astype(np.single)
        else:
            print("unsupported data format: ", file_path)
            exit(-1)
            
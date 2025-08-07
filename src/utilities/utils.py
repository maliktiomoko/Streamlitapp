import time
import os

def time_execution(func, *args, **kwargs):
    """Measure the execution time of a function and return its result with the duration."""
    start_time = time.time()
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()
    
    duration = end_time - start_time
    return result, duration

def log(message):
    """Prints a formatted log message."""
    print(f"\n📝 {message}\n" + "-" * 50)


def ensure_3d(array):
    """Ensure the input array is 3D by adding a dimension of size 1 if it is 1D or 2D."""
    if array.ndim == 1:
        return array.reshape(array.shape[0], 1, 1)
    elif array.ndim == 2:
        return array.reshape(array.shape[0], array.shape[1], 1)
    elif array.ndim == 3:
        return array
    else:
        raise ValueError("Input array must be 1D, 2D, or 3D.")
    
def save_uploaded_files(uploaded_files, folder="temp_data"):
    os.makedirs(folder, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    return file_paths
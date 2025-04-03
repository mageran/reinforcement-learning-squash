import tensorflow as tf
import time

def matrix_mul():
    # Create a large matrix
    matrix_size = 10000
    matrix = tf.random.normal((matrix_size, matrix_size))
    start_time = time.time()
    result = tf.matmul(matrix, matrix)
    end_time = time.time()
    print(f"time: {end_time - start_time}")


def run_test():
    # Create a large matrix
    matrix_size = 10000
    matrix = tf.random.normal((matrix_size, matrix_size))

    # GPU test
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            start_gpu = time.time()
            result_gpu = tf.matmul(matrix, matrix)
            end_gpu = time.time()
            print(f"GPU time: {end_gpu - start_gpu}")

        # CPU test
        with tf.device('/CPU:0'):
            start_cpu = time.time()
            result_cpu = tf.matmul(matrix, matrix)
            end_cpu = time.time()
            print(f"CPU time: {end_cpu - start_cpu}")
    else:
        print("No GPUs found, running on CPU")
        start_cpu = time.time()
        result_cpu = tf.matmul(matrix, matrix)
        end_cpu = time.time()
        print(f"CPU time: {end_cpu - start_cpu}")

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU available:", gpus)
    else:
        print("GPU not available.")


if __name__ == "__main__":
    main()

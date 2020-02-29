import pandas

def generate(batch_size):
    """
    generate batches of examples
    :param batch_size: the size of the batch
    :return: yield a dictionary {"stars": np.array(shape=batch_size), "id": np.array(shape=batch_size), "review": np.array(shape=batch_size, max_len)}
    """
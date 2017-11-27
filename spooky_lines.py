import tensorflow as tf


def file_input_ops(filename):
    """Function for processing input csv files

    Args:
        filename: String csv file name

    Returns:
        A tuple of feature, label
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(filename_queue)

    record_defaults = [[''], [''], [0], [0], [0]]
    example_id, sentence, eap, hpl, mws = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.stack([example_id, sentence])
    author = tf.stack([eap, hpl, mws])

    return features, author

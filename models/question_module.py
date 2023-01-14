import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def build_question_module(query_tensor, inputQueryLengths, recurrentCellSize, input_p, output_p):
    """
    Builds the question module for the neural network.

    Args:
        query_tensor (tf.Tensor): Placeholder for questions.
        inputQueryLengths (tf.Tensor): Placeholder for question lengths.
        recurrentCellSize (int): GRU cell size.
        input_p (float): Input dropout keep probability.
        output_p (float): Output dropout keep probability.

    Returns:
        tf.Tensor: Outputs of the question module.
    """
    queryGRU = tf.nn.rnn_cell.GRUCell(recurrentCellSize)
    GRUdrop = tf.nn.rnn_cell.DropoutWrapper(queryGRU, input_keep_prob=input_p, output_keep_prob=output_p)
    questionModuleOutputs, _ = tf.nn.dynamic_rnn(GRUdrop, query_tensor, dtype=tf.float32, scope="questionModule")
    qslice = tf.gather_nd(questionModuleOutputs, inputQueryLengths)
    return qslice
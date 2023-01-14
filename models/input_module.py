import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def build_input_module(context_tensor, inputSentenceEnd, recurrentCellSize, input_p, output_p):
    """
    Builds the input module for the neural network.

    Args:
        context_tensor (tf.Tensor): Placeholder for context.
        inputSentenceEnd (tf.Tensor): Placeholder for sentence ends.
        recurrentCellSize (int): GRU cell size.
        input_p (float): Input dropout keep probability.
        output_p (float): Output dropout keep probability.

    Returns:
        tf.Tensor: Outputs of the input module.
    """
    inputGRU = tf.nn.rnn_cell.GRUCell(recurrentCellSize)
    GRUdrop = tf.nn.rnn_cell.DropoutWrapper(inputGRU, input_keep_prob=input_p, output_keep_prob=output_p)
    inputModuleOutputs, _ = tf.nn.dynamic_rnn(GRUdrop, context_tensor, dtype=tf.float32, scope="inputModule")
    gslice = tf.gather_nd(inputModuleOutputs, inputSentenceEnd)
    return gslice
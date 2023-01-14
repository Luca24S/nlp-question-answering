import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def build_answer_module(memory, question, Dim):
    """
    Builds the answer module for the network.

    Args:
        memory (tf.Tensor): Final memory state from the episodic memory module.
        question (tf.Tensor): Final question vector.
        Dim (int): Dimension of the output vector.

    Returns:
        tf.Tensor: Logits representing predicted answer scores.
    """
    with tf.variable_scope("answer"):
        concatenated_memory = tf.concat([memory, question], axis=-1)
        W = tf.get_variable("W", [concatenated_memory.get_shape()[-1], Dim], initializer=tf.random_normal_initializer())
        logits = tf.matmul(concatenated_memory, W)
        return logits
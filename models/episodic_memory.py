import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def attention_mechanism(context, memory, question, existing_facts, recurrentCellSize):
    """
    Implements the attention mechanism for episodic memory.

    Args:
        context (tf.Tensor): Context tensor (facts).
        memory (tf.Tensor): Memory tensor (current memory state).
        question (tf.Tensor): Question tensor.
        existing_facts (tf.Tensor): Binary mask for existing facts.
        recurrentCellSize (int): GRU cell size.

    Returns:
        tf.Tensor: Attention weights.
    """
    with tf.variable_scope("attention"):
        concat_features = tf.concat(
            [context, memory, question, context * question, context * memory, 
             (context - question) ** 2, (context - memory) ** 2],
            axis=-1
        )
        
        W1 = tf.get_variable("W1", [recurrentCellSize * 7, recurrentCellSize], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("b1", [recurrentCellSize], initializer=tf.zeros_initializer())
        W2 = tf.get_variable("W2", [recurrentCellSize, 1], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

        hidden = tf.nn.relu(tf.tensordot(concat_features, W1, axes=1) + b1)
        attention_logits = tf.tensordot(hidden, W2, axes=1) + b2

        attention_weights = tf.nn.softmax(attention_logits * existing_facts, axis=1)
        return attention_weights

def build_episodic_memory(context, question, steps, recurrentCellSize, existing_facts):
    """
    Builds the episodic memory module for the network.

    Args:
        context (tf.Tensor): Context tensor.
        question (tf.Tensor): Question tensor.
        steps (int): Number of steps in episodic memory.
        recurrentCellSize (int): GRU cell size.
        existing_facts (tf.Tensor): Binary mask for existing facts.

    Returns:
        tf.Tensor: Final memory state.
    """
    memory = question
    with tf.variable_scope("episodic_memory"):
        for _ in range(steps):
            attention_weights = attention_mechanism(context, memory, question, existing_facts, recurrentCellSize)
            context_gru = tf.nn.rnn_cell.GRUCell(recurrentCellSize)
            attended_context = tf.reduce_sum(context * attention_weights, axis=1)
            memory = context_gru(attended_context, memory)[0]
    return memory
import numpy as np

def batch_preparation(batch_data):
    """
    Prepares a batch of data for feeding into the neural network.

    Args:
        batch_data (list): List of tuples with context, question, and answer.

    Returns:
        dict: Dictionary of placeholders and their respective data.
    """
    context_vec, sentence_ends, questionVect, _, _, answer_vector, _ = zip(*batch_data)
    
    # Pad sentence ends
    max_length = max(len(seq) for seq in sentence_ends)
    padded_sentence_ends = np.zeros((len(sentence_ends), max_length, 2), dtype=np.int32)
    for i, ends in enumerate(sentence_ends):
        padded_sentence_ends[i, :len(ends), 1] = np.array(ends) - 1
        padded_sentence_ends[i, :, 0] = i
    
    # Pad context vectors
    max_context_length = max(len(context) for context in context_vec)
    context_dim = context_vec[0].shape[1]
    padded_contexts = np.zeros((len(context_vec), max_context_length, context_dim), dtype=np.float32)
    for i, context in enumerate(context_vec):
        padded_contexts[i, :len(context), :] = context
    
    # Pad questions
    max_question_length = max(len(q) for q in questionVect)
    question_dim = questionVect[0].shape[1]
    padded_questions = np.zeros((len(questionVect), max_question_length, question_dim), dtype=np.float32)
    for i, question in enumerate(questionVect):
        padded_questions[i, :len(question), :] = question
    
    # Prepare query lengths
    query_lengths = np.array([[i, len(q) - 1] for i, q in enumerate(questionVect)])
    
    return {
        "context": padded_contexts,
        "sentence_ends": padded_sentence_ends,
        "query": padded_questions,
        "query_lengths": query_lengths,
        "gold_standard": answer_vector,
    }
import tensorflow.compat.v1 as tf
import numpy as np
from pathlib import Path
from utils.preprocess import GloVe_extractData, find_context 
from utils.batch_utils import batch_preparation               
from models.input_module import build_input_module
from models.question_module import build_question_module
from models.episodic_memory import build_episodic_memory
from models.answer_module import build_answer_module

# === Load data ===
glove_path = Path("data/glove.6B/glove.6B.50d.txt")
babi_path = Path("data/tasks_1-20_v1-2/en/")
train_file = babi_path / "qa5_three-arg-relations_train.txt"
test_file = babi_path / "qa5_three-arg-relations_test.txt"

print("Loading GloVe embeddings...")
gloveDict, gloveVectors = GloVe_extractData(glove_path)

print("Loading bAbI datasets...")
train_data = find_context(train_file, gloveDict, gloveVectors)
test_data = find_context(test_file, gloveDict, gloveVectors)

# === Define Hyperparameters ===
recurrentCellSize = 128
input_p, output_p = 0.5, 0.5
steps = 4  # Number of steps in episodic memory
Dim = 50  # GloVe embedding dimension
batch_size = 256
learning_rate = 0.01
training_iterations = 10000
display_step = 100

# === Placeholders ===
context_tensor = tf.placeholder(tf.float32, [None, None, Dim], name="context")
inputSentenceEnd = tf.placeholder(tf.int32, [None, None, 2], name="sentence_ends")
query_tensor = tf.placeholder(tf.float32, [None, None, Dim], name="query")
inputQueryLengths = tf.placeholder(tf.int32, [None, 2], name="query_lengths")
gold_standard = tf.placeholder(tf.float32, [None, Dim], name="gold_standard")
existing_facts = tf.placeholder(tf.float32, [None, None, 1], name="existing_facts")

# === Build Model ===
print("Building model...")

# Input Module
gslice = build_input_module(context_tensor, inputSentenceEnd, recurrentCellSize, input_p, output_p)

# Question Module
qslice = build_question_module(query_tensor, inputQueryLengths, recurrentCellSize, input_p, output_p)

# Episodic Memory Module
final_memory = build_episodic_memory(gslice, qslice, steps, recurrentCellSize, existing_facts)

# Answer Module
logits = build_answer_module(final_memory, qslice, Dim)

# Loss and Optimization
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gold_standard, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# Accuracy
predictions = tf.argmax(logits, axis=1)
true_answers = tf.argmax(gold_standard, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, true_answers), tf.float32))

# === Training and Evaluation ===
print("Starting session...")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def train_model(sess, train_data, test_data, batch_size, iterations, display_step):
    """
    Trains the model on the training data and evaluates on the test data.

    Args:
        sess (tf.Session): TensorFlow session.
        train_data (list): Training dataset.
        test_data (list): Testing dataset.
        batch_size (int): Batch size for training.
        iterations (int): Number of training iterations.
        display_step (int): Frequency of displaying training metrics.
    """
    for i in range(iterations):
        # Training
        batch = train_data[np.random.randint(len(train_data), size=batch_size)]
        train_feed_dict = batch_preparation(batch)  # Use batch_preparation from utils/
        sess.run(train_op, feed_dict=train_feed_dict)

        if i % display_step == 0:
            # Calculate training loss and accuracy
            train_loss, train_acc = sess.run([loss, accuracy], feed_dict=train_feed_dict)
            print(f"Iteration {i}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

            # Evaluate on test data
            eval_batch = test_data[np.random.randint(len(test_data), size=batch_size)]
            eval_feed_dict = batch_preparation(eval_batch)  # Use batch_preparation from utils/
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict=eval_feed_dict)
            print(f"Iteration {i}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print("Training complete.")

# Train the model
train_model(sess, train_data, test_data, batch_size, training_iterations, display_step)

# Evaluate on the full test set
print("Evaluating on test set...")
test_feed_dict = batch_preparation(test_data)  # Use batch_preparation from utils/
final_test_acc = sess.run(accuracy, feed_dict=test_feed_dict)
print(f"Final Test Accuracy: {final_test_acc:.4f}")

sess.close()
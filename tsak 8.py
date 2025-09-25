import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.tag import PerceptronTagger
from nltk.tag import tnt
from nltk.tag import brill, brill_trainer

# Download required NLTK data
nltk.download('treebank')

# Load and prepare the corpus
corpus = list(treebank.tagged_sents())

# Split into training and test sets
split_index = int(0.8 * len(corpus))
train_data = corpus[:split_index]
test_data = corpus[split_index:]

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Train HMM tagger
print("\nTraining HMM Tagger...")
hmm_tagger = hmm.HiddenMarkovModelTrainer().train(train_data)
hmm_accuracy = hmm_tagger.evaluate(test_data)
print(f"HMM Tagger Accuracy: {hmm_accuracy:.4f}")

# Train Perceptron Tagger (modern alternative to MaxEnt)
print("\nTraining Perceptron Tagger...")
perceptron_tagger = PerceptronTagger(load=False)
perceptron_tagger.train(train_data)
perceptron_accuracy = perceptron_tagger.evaluate(test_data)
print(f"Perceptron Tagger Accuracy: {perceptron_accuracy:.4f}")

# Train TnT Tagger (another statistical tagger)
print("\nTraining TnT Tagger...")
tnt_tagger = tnt.TnT()
tnt_tagger.train(train_data)
tnt_accuracy = tnt_tagger.evaluate(test_data)
print(f"TnT Tagger Accuracy: {tnt_accuracy:.4f}")

# Test all taggers on a sample sentence
test_sentence = "The quick brown fox jumps over the lazy dog".split()

print("\n" + "="*50)
print("PREDICTIONS ON SAMPLE SENTENCE:")
print("="*50)

hmm_prediction = hmm_tagger.tag(test_sentence)
perceptron_prediction = perceptron_tagger.tag(test_sentence)
tnt_prediction = tnt_tagger.tag(test_sentence)

print(f"HMM Prediction: {hmm_prediction}")
print(f"Perceptron Prediction: {perceptron_prediction}")
print(f"TnT Prediction: {tnt_prediction}")

# Compare results
print("\n" + "="*50)
print("ACCURACY COMPARISON:")
print("="*50)
print(f"HMM Tagger: {hmm_accuracy:.4f} ({hmm_accuracy*100:.2f}%)")
print(f"Perceptron Tagger: {perceptron_accuracy:.4f} ({perceptron_accuracy*100:.2f}%)")
print(f"TnT Tagger: {tnt_accuracy:.4f} ({tnt_accuracy*100:.2f}%)")

# Find the best tagger
accuracies = {
    "HMM": hmm_accuracy,
    "Perceptron": perceptron_accuracy,
    "TnT": tnt_accuracy
}
best_tagger = max(accuracies, key=accuracies.get)
print(f"\nBest performing tagger: {best_tagger} ({accuracies[best_tagger]:.4f})")

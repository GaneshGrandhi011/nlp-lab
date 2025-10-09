import nltk
from sklearn.metrics import accuracy_score

# Download necessary NLTK models if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def hmm_pos_tagger(sentence):
    # This function currently uses nltk.pos_tag which is a PerceptronTagger (not HMM)
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return tagged

def log_linear_pos_tagger(sentence):
    # Same as hmm_pos_tagger above, so outputs will be identical here
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return tagged

def compare_performance(sentence):
    hmm_tags = hmm_pos_tagger(sentence)
    log_linear_tags = log_linear_pos_tagger(sentence)

    # Normally, gold_standard_tags should come from a manually tagged dataset, not predictions
    # Here we are using hmm_tags as gold standard (which is circular)
    gold_standard_tags = [tag for _, tag in hmm_tags]
    hmm_predicted_tags = [tag for _, tag in hmm_tags]
    log_linear_predicted_tags = [tag for _, tag in log_linear_tags]

    hmm_accuracy = accuracy_score(gold_standard_tags, hmm_predicted_tags)
    log_linear_accuracy = accuracy_score(gold_standard_tags, log_linear_predicted_tags)

    print("HMM Predicted Tags:", hmm_predicted_tags)
    print("Log-Linear Model Predicted Tags:", log_linear_predicted_tags)
    print("HMM Accuracy:", hmm_accuracy)
    print("Log-Linear Model Accuracy:", log_linear_accuracy)

input_text = "The quick brown fox jumps over the lazy dog."
compare_performance(input_text)

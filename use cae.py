from transformers import pipeline

MODEL_PATH = "SamLowe/roberta-base-go_emotions"

def load_classifier_model(model_name):
    """Loads the pre-trained text classification model from Hugging Face."""
    print(f"Initializing emotion detection model: {model_name}...")
    print("This may take a moment on the first run as the model is downloaded.")
    
    # Create a text-classification pipeline with the specified model
    emotion_classifier = pipeline("text-classification", model=model_name, top_k=1)
    
    print("Model loaded successfully. You can start analyzing text now.")
    return emotion_classifier

def display_result(text, result):
    """Formats and prints the emotion analysis result."""
    prediction = result[0]
    label = prediction['label']
    score = prediction['score']
    
    print("\n-- Analysis Result --")
    print(f"Text Input: '{text}'")
    print(f"Predicted Emotion: {label.capitalize()} (Confidence: {score:.2%})")

def main():
    """Main function to run the interactive emotion analysis tool."""
    print("====================================")
    print(" Interactive Emotion Analysis tool")
    print("====================================")
    print("Type 'exit' or 'quit' to close the Program.\n")

    # Load the model
    classifier = load_classifier_model(MODEL_PATH)

    # Start the interactive loop
    while True:
        user_input = input("\nEnter text to Analyze: ")

        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the tool. Goodbye!")
            break

        # Check for empty input
        if not user_input.strip():
            print("Please enter some text to analyze.")
            continue

        # Get the prediction from the model
        prediction_result = classifier(user_input)
        
        # Display the formatted result
        display_result(user_input, prediction_result)

if __name__ == "__main__":
    main()

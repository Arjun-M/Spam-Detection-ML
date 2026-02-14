from model import run_model


def main():
    print("Spam/Scam Message Detector")
    print("Type a message to classify, or type 'quit' to exit.\n")

    while True:
        user_input = input("Enter message: ")

        if user_input.strip().lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        output = run_model(user_input)

        if not output["ok"]:
            print(f"Invalid input: {output['error']}\n")
            continue

        result = output["result"]
        print(f"Spam: {result['is_spam']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Category: {result['category']}\n")


if __name__ == "__main__":
    main()

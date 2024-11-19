from whisper_tools import WhisperTools

def main():
    print("Starting model download...")
    tools = WhisperTools()
    tools.download_models()
    print("Model download complete!")

if __name__ == "__main__":
    main()
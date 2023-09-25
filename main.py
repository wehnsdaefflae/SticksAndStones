def comment_appearance(image: str) -> str:
    # scrape_bing_chat
    # or vilt
    pass


def respond(input_text: str) -> str:
    # chatgpt
    pass


def say(text: str) -> None:
    # bark
    pass


def listen() -> str:
    # whisper
    pass


def main() -> None:
    while True:
        image = ""
        response = comment_appearance(image)
        while True:
            say(response)
            input_text = listen()
            response = respond(input_text)
            if len(input_text) < 1:
                break


if __name__ == '__main__':
    main()

import random

components = [
    [
        "give me a ride home",
        "share that recipe with me",
        "water the plants",
        "give this letter to Jane",
        "assist me with this package",
        "teach me that technique",
        "introduce me to your friend",
        "let me know your thoughts",
        "share your umbrella",
        "help set the table",
        "walk the dog for me",
        "turn off the lights",
        "save some dessert for me",
        "give me a call tomorrow",
        "remind me of the meeting",
        "lend me your book",
        "show me how to use this",
        "pick me up from the station",
        "keep an eye on my bag"
    ],
    [
        "for a few minutes",
        "while I'm away",
        "until I return",
        "by this evening",
        "for the next week",
        "whenever you can",
        "before you leave",
        "just this once",
        "only if it's convenient",
        "as soon as possible",
        "if you have time later",
        "during the break",
        "after dinner",
        "in case I forget",
        "every other day",
        "on your way out",
        "in the next hour",
        "by the end of the day",
        "the next time you see her",
        "if you're going that way"
    ],
    [
        "because I'm not feeling well",
        "since it's on sale today",
        "as it's crucial for the presentation",
        "considering the rain",
        "given our previous discussion",
        "due to the early deadline",
        "knowing your expertise in the area",
        "seeing that you're closer",
        "to surprise our colleague",
        "with the weather being so cold",
        "keeping in mind the event tomorrow",
        "in light of the recent changes",
        "as I'm out of options",
        "being the last one left",
        "to ensure it's safe",
        "since you've done it before",
        "as I trust your judgment",
        "given its sentimental value",
        "in preparation for the trip",
        "with the party coming up"
    ]
]


def generate_sentence() -> str:
    sentence = ""
    for component in components:
        sentence += random.choice(component) + " "
    return sentence


if __name__ == "__main__":
    print(generate_sentence())

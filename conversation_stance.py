import random
from typing import List


def generate_random_stance() -> str:
    # Define the categories with concise and unambiguous descriptions in the second person
    emotion: List[str] = [
        # "You're full of joy",
        # "You're indifferent",
        # "You're deeply upset"
        "You're consumed by anger",
        "You're overwhelmed with sadness",
        "You're paralyzed by fear",
        "You're bubbling with jealousy",
        "You're seething with disgust"
    ]

    engagement: List[str] = [
        "you share personal stories",
        "you stick to facts",
        "you're always asking questions"
    ]

    consistency_clarity: List[str] = [
        "you're firm in your views",
        "you change topics quickly",
        "you're hard to read"
    ]

    # Randomly select a description from each category
    random_emotion: str = random.choice(emotion)
    random_engagement: str = random.choice(engagement)
    random_consistency_clarity: str = random.choice(consistency_clarity)

    # Construct the final description
    stance_description: str = f"{random_emotion}. While {random_engagement}, {random_consistency_clarity}."

    return stance_description


# Testing the function
if __name__ == "__main__":
    for _ in range(10):
        print(generate_random_stance())

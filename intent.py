"""_summary_
tag = string
patterns = list of string/ questios
response = can be a function that returns string or a string
"""
def respond_helloworld() -> str:
    return "Hello World, Goodbye"


Intents = [
    {
        "tag": "greeting",
        "patterns": [
            "Hi",
            "Hey",
            "How are you",
            "Is anyone there?",
            "Hello",
            "Good day"
            ],
        "response": [
            "Hey :-)",
            "Hello, thanks for visiting",
            "Hi there, what can I do for you?",
            "Hi there, how can I help?"
            ]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye"],
        "response": respond_helloworld
    },
    {
        "tag": "thanks",
        "patterns": ["Thanks", "Thank you", "That's helpful", "Thank's a lot!"],
        "response": ["Happy to help!", "Any time!", "My pleasure"]
    },
    {
        "tag": "ceb_greeting",
        "patterns": [
            "Kamusta",
            "Oy",
            "Kamusta ka?",
            "Naay tawo dinhi?",
            "Hello",
            "Maayong adlaw"
            ],
        "response": [
            "Oy :-)",
            "Hello, salamat sa pagbisita",
            "Hi diha, unsa man akong mahimo alang nimo?",
            "Hi diha, unsaon kong matabang?"
            ]
    },
    {
        "tag": "ceb_goodbye",
        "patterns": ["Bye", "Sunod napud", "Goodbye"],
        "response": "Sigee byeee 👋"
    },
    {
        "tag": "ceb_thanks",
        "patterns": ["Salamat", "Salamat kaayo"],
        "response": ["Malipayon ko nga nakatabang!", "Sa bisan unsang oras!", "Salamat kay nakatabang ko :->"]
    },
]
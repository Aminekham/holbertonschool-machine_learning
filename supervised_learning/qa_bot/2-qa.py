#!/usr/bin/env python3
question_answer = __import__('0-qa').question_answer
"""

"""
def answer_loop(reference):
    """

    """
    while(1):
        x = input("Q:")
        x = x.lower()
        if x == "exit" or x == "quit" or x == "goodbye" or x == "bye":
            print("A: Goodbye")
            break
        if question_answer(x, reference) == None:
            print("A: Sorry, I do not understand your question.")
            continue
        print("A:", question_answer(x, reference))

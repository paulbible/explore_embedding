"""
    Some examples
"""

# defining a function
def plus_one(x):
    # x is 5 during the call
    return x + 1


def main():
    print('hello')
    y = 10

    # calling a function
    # f(x) = x*x,   f(2) -> 4
    # input -> output processes.
    z = plus_one(5)
    print(z)
    print(y)

    text = "this is some text. how are you doing. Where is the car? car car is where where core core"
    words = text.split()
    print(words)
    word_count = {}
    for word in words:
        print(word)
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    print(word_count)



main()

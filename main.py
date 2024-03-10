from model_source import *

OPERATORS = [XOR, IMP, BIC]
DATA = [(0, 0),
        (0, 1),
        (1, 0),
        (1, 1)]

# basic operators
print("basic operators -----")

for operator in OPERATORS:
    for (a, b) in DATA:
        result = 0 if operator(a, b) < 0.5 else 1
        print(f"{operator.__name__}({a}, {b}) \t= {result}")
        #print(f"(symbols){XOR({a}, {b}) = {a ^ b}")

print("\nadvanced function-----")

a = 0
b = 1
c = 0
result = 0 if AND(XOR(a, BIC(b, c)), c) < 0.5 else 1
print(f"AND(XOR({a}, BIC({b}, {c})), {c}) = {result}")

from typing import TypedDict


# only used for getting hints from editor,no check ,no validation,
class Person(TypedDict):
    name: str
    age: int


new_person: Person = {"name": "kristal", "age": 35}
new_person1: Person = {"name": "kristal", "age": "38"}

print(new_person)
print(new_person1)

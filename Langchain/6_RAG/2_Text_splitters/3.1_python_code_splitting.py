import chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""


# initialize the splitters
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=288,  # hit and trial from this website:https://chunkviz.up.railway.app/
    chunk_overlap=0,
)


# perform splitting

chunks = splitter.split_text(text)
print(len(chunks))  # 2
print(chunks)
"""Output:
['class Student:\n    def __init__(self, name, age, grade):\n        self.name = name\n        self.age = age\n        self.grade = grade  # Grade is a float (like 8.5 or 9.2)\n\n    def get_details(self):\n        return self.name"\n\n    def is_passing(self):\n        return self.grade >= 6.0', '# Example usage\nstudent1 = Student("Aarav", 20, 8.2)\nprint(student1.get_details())\n\nif student1.is_passing():\n    print("The student is passing.")\nelse:\n    print("The student is not passing.")']
"""
print(type(chunks))  # <class 'list'>
print(chunks[0])
"""first chunk Output:
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0
"""
print(type(chunks[0]))  # <class 'str'>

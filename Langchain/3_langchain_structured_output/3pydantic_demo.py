from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "Kristal"  # default value
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description="A decimal value representing the cgpa of the student",
    )


# new_student = {"name": "kristal"}
# new_student = {"name": 12} # error ,data validation

# default value
new_student = {"age": "32", "email": "abc@gmail.com"}
student = Student(**new_student)
print(student)
print(type(student))
# print(student["name"])  # error
print(student.name)  # this will work
print(student.age)


student_dict = student.model_dump()
print(student_dict["age"])
print(type(student_dict))


student_json = student.model_dump_json()
print(type(student_json))
import json

student_dict_from_json = json.loads(student_json)
print(student_dict_from_json["age"])

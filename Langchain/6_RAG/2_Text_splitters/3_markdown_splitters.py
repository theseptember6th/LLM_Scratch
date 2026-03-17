from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

# initialize the splitter

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=350,  # hit and trial from this website:https://chunkviz.up.railway.app/
    chunk_overlap=0,
)


# perform the split
chunks = splitter.split_text(text)


print(len(chunks))  # 2
print(chunks)
"""Output:['# Project Name: Smart Student Tracker\n\nA simple Python-based project to manage and track student data, including their grades, age, and academic status.\n\n\n## Features\n\n- Add new students with relevant info\n- View student details\n- Check if a student is passing\n- Easily extendable class-based design', '## 🛠 Tech Stack\n\n- Python 3.10+\n- No external dependencies\n\n\n## Getting Started\n\n1. Clone the repo  \n   ```bash\n   git clone https://github.com/your-username/student-tracker.git']"""

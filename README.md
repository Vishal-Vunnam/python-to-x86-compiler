
## 🛠 Build & Run Instructions

1. **Build the runtime system (if not already built):**
   ```bash
   make -C runtime
Build the compiler:

bash
Copy
Edit
make
Compile a test Python program:

bash
Copy
Edit
./pyyc mytests/test1.py
Link the generated assembly with the runtime:

bash
Copy
Edit
gcc -m32 -g mytests/test1.s runtime/libpyyruntime.a -lm -o mytests/test1
Run the compiled binary:

bash
Copy
Edit
cat mytests/test1.in | mytests/test1 > mytests/test1.out
Generate the expected output using Python 3.10:

bash
Copy
Edit
cat mytests/test1.in | python3.10 mytests/test1.py > mytests/test1.expected
Compare your program's output:

bash
Copy
Edit
diff -w -B -u mytests/test1.expected mytests/test1.out
You can automate some of these steps using make targets included in Test.mk.

## 🧪 Automated Testing
A testing framework is included using pytest, which makes it easy to validate the compiler across a wide range of test cases.

Run tests on your own suite:

bash
Copy
Edit
cd tests
pytest --pyyctests mytests
Run tests on pre-written groups (e.g., autograde/p0/easy):

bash
Copy
Edit
cd tests
pytest --pyyctests autograde
Tests are organized by difficulty and feature coverage to aid iterative development.

## 💡 Goals of This Project
Implement a full compiler pipeline from Python to x86

Explore register allocation, control flow graphs, and interference graphs

Understand low-level code generation and runtime system interaction

Build a system that can be tested against real programs

## 📦 Environment Setup
To isolate dependencies, consider using pipenv or venv to manage Python versions and required packages.

Let me know if you’d like this adapted for a personal website, technical blog, or portfolio project!

yaml
Copy
Edit

---

Just copy and paste the above Markdown into your `README.md` file. If you want any badges, project screenshots, or GIF demos added, I can help with that too.









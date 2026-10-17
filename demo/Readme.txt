python -m venv .myenv

myenv\Script\activate

pip install fastapi uvicorn

uvicorn main:app --reload

pip freeze > requirements.txt


GIT Commands:
-------------

git init

Add remote origin (Taken from git repo)

git branch -M main

git add .

git commit -m "v1"

git push -u origin main
# Dynamic Student Session Lab

Flask website for running a real adaptive MCQ episode with the trained dynamic DQN model.

Students can:

- sign up with a unique name
- receive a saved ID such as `00001`
- sign in later with name + ID
- play an adaptive episode selected by the trained model
- save the final effective ability in SQLite
- start the next episode from the stored effective ability
- qualify for `A+` or `Golden A+` after reaching effective ability `30`

## Run locally

```bash
python3 app.py
```

Open:

```text
http://127.0.0.1:5000
```

The app uses SQLite at:

```text
data/student_sessions.db
```

## Model file

The original `final_model.pt` checkpoint is about 211 MB because it includes training data. This project can use a smaller deployment checkpoint at:

```text
models/final_model_slim.pt
```

If you want to use another model path:

```bash
MODEL_PATH="/path/to/final_model.pt" python3 app.py
```

On Render, set `MODEL_PATH` only if your model is not inside `models/final_model_slim.pt`.

## Deploy on Render

Use these settings:

```text
Build command: pip install -r requirements.txt
Start command: gunicorn --workers 1 --timeout 180 app:app
```

If your GitHub repo contains the app inside a folder named `Real_Student_Session_lab`, set:

```text
Root Directory: Real_Student_Session_lab
```

If `app.py`, `templates/`, and `static/` are at the GitHub repo root, leave Root Directory blank.

The design is loaded through Flask `url_for('static', ...)`, so Render must deploy the folder that contains both `app.py` and `static/`.

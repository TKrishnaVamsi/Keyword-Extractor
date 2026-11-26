# KeyLens frontend

A static HTML/CSS/JS frontend for the Keyword Extraction FastAPI project.

## Local test
1. Start the API in one terminal:
   `python -m uvicorn src.api:app --host 127.0.0.1 --port 8000`
2. In a second terminal, go into this folder and run:
   `python -m http.server 5500`
3. Open `http://127.0.0.1:5500`.
4. Leave Backend API URL as `http://127.0.0.1:8000`.

## Enable CORS in FastAPI
Add this import near the top of `src/api.py`:

```python
from fastapi.middleware.cors import CORSMiddleware
```

After the line where your FastAPI app is created (`app = FastAPI(...)`), add:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://YOUR-VERCEL-DOMAIN.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Replace the Vercel domain with the exact deployed frontend origin. For quick temporary testing only, `allow_origins=["*"]` can be used, but explicit origins are better. Restart FastAPI after changing this.

## Deploy the frontend to Vercel
1. Add this `frontend` folder to your existing GitHub project.
2. Commit and push it:
   `git add frontend`
   `git commit -m "Add keyword extraction web interface"`
   `git push`
3. In Vercel, choose **Add New → Project** and import your GitHub repository.
4. Set **Root Directory** to `frontend`.
5. Choose **Other** as framework preset. This is a static site; no build command is needed and the output directory is `.`.
6. Click **Deploy**.
7. Copy the deployed `https://....vercel.app` URL.
8. Add that exact URL to the FastAPI CORS `allow_origins` list, commit and push the backend change, then wait for the API to redeploy.
9. Open the Vercel site. In **Backend API URL**, enter your public FastAPI URL (for example, `https://your-api.onrender.com`) and click **Check API connection**.

You can also deploy from this folder with the Vercel CLI: run `npx vercel` from inside `frontend`.

## Deploy the FastAPI backend
Create a separate Render **Web Service** from the same GitHub repository:
- Root Directory: leave blank if `requirements.txt` and `src/` are at repository root.
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`

Before deploying, verify that the model registry and artifact store used by `MLFLOW_TRACKING_URI` are available in the Render environment. This project currently uses a local `mlruns` directory and a local MLflow registry. A fresh Render instance will not automatically have your laptop's MLflow database/artifacts. For a quick demo, the model artifacts and MLflow metadata must be present in the deployed repository/runtime at the expected paths; for a reliable deployment, use remote durable MLflow tracking/artifact storage and configure the service to use it. Do not commit `.venv/`, caches, or secrets.

Render free web services can spin down when idle, so the first API request after inactivity may take time. The SentenceTransformer model also needs sufficient memory and may download its Hugging Face weights at startup.

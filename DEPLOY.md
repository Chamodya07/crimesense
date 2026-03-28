# Deploy

## Prerequisites

- Install and authenticate the Google Cloud CLI.
- Install and authenticate the Firebase CLI.
- Make sure the correct Google Cloud project is selected in both CLIs.

## Cloud Run

Deploy the Streamlit app from the repo root:

```bash
gcloud run deploy crimesense-streamlit --source . --region us-central1 --allow-unauthenticated
```

This repo includes a `Dockerfile` that starts:

```bash
streamlit run app/streamlit_app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true
```

## Firebase Hosting

Initialize Hosting from the repo root and use the existing `public/` directory when prompted:

```bash
firebase init hosting
```

Deploy Hosting:

```bash
firebase deploy --only hosting
```

## Notes

- `firebase.json` rewrites all Hosting routes to the Cloud Run service `crimesense-streamlit` in `us-central1`.
- `public/index.html` is a minimal placeholder required for Hosting.
- `requirements.txt` was left intact; no existing dependencies were removed.

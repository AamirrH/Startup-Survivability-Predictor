# Canopy — Random Forest startup web app

The finalized Random Forest model now has a responsive Flask interface. All application code, styles, model artifacts, tests, and deployment files are contained in `webapp/`.

## Run locally

From the repository root, using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r webapp/requirements.txt
.\.venv\Scripts\python.exe -m webapp.app
```

Open **http://127.0.0.1:8000**. If the project environment already exists, skip the first command. Waitress serves the app without debug mode. The saved model is included, so training is not needed at startup.

The training data is a historical snapshot ending in 2015. The model compares a new profile with older companies that were acquired or closed; it does not prove that a startup will survive or succeed. The app accepts valid founded and funding years after 2015, but shows a warning because newer startups were not part of model validation. It rejects future years and inconsistent timelines, such as first funding after last funding. A funding-round count itself is not limited to 2015.

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

## Use it on a phone

For a local demonstration, connect the computer and phone to the same Wi-Fi network. Start the server with a LAN-accessible binding:

```powershell
$env:HOST = "0.0.0.0"
$env:PORT = "8000"
.\.venv\Scripts\python.exe -m webapp.app
```

Find the computer's Wi-Fi IPv4 address using `ipconfig`, then open `http://YOUR_COMPUTER_IP:8000` in the phone browser. `127.0.0.1` on a phone refers to the phone itself. A trusted private-network firewall rule may be needed; do not disable the firewall. Local network availability depends on Wi-Fi isolation and firewall settings. For access outside that Wi-Fi, deploy to a hosting service with HTTPS.

## Public deployment

The app is deployment-ready; a public hosting service has not been provisioned by this build. No hosting account or credentials are embedded.

### Recommended choice for this project

Deploy the whole app and model together as one Render Web Service. The Flask UI and prediction endpoint stay in the same process, and the saved `random_forest.joblib` remains a versioned artifact beside the code. This is the simplest setup for a college demo: one repository, one URL, one deployment, and no frontend/backend coordination. Render supports Python web services and provides a unique `onrender.com` URL. Its Free service is suitable for testing and hobby projects but spins down after 15 minutes of inactivity, so the first request after idle can take about a minute. See [Render's Flask guide](https://render.com/docs/deploy-flask) and [free-service limits](https://render.com/docs/free).

For Render, use the repository root as the service root, `pip install -r webapp/requirements.txt` as the build command, and `python -m webapp.app` as the start command. Set `HOST=0.0.0.0`; Render supplies `PORT`. The saved model does not need a separate hosting service. Keep it in `webapp/artifacts/random_forest.joblib` and deploy it with the app. Do not put the model in a public model hub unless you specifically want independent model versioning or a separate inference API.

### Other sensible options

| Option | Deploy | Model location | Best use | Tradeoff |
| --- | --- | --- | --- | --- |
| Render Web Service | Entire `webapp/` app | Bundled `webapp/artifacts/` | Easiest free demo | Sleeps when idle on Free |
| Railway | Entire app or the included Dockerfile | Bundled in the image | Simple small service | Usage-based billing; Hobby is paid |
| Hugging Face Spaces | Docker or Gradio version | Bundled or downloaded at build time | ML-focused demos | This Flask app needs a Space adapter |
| Cloud Run / Azure App Service / AWS App Runner | Docker image | Bundled in image or object storage | More production control | More setup and cloud billing |
| VPS | Docker or Waitress behind HTTPS | Bundled on the server | Full control and stable uptime | You manage updates, TLS, and monitoring |

Railway supports Flask directly or with Docker, but uses a paid usage model after its trial. Hugging Face Spaces supports Docker, Gradio, and static HTML; this existing Flask app is best kept on a general Python service unless you intentionally convert it to Gradio. Keep the model and preprocessing pipeline together: separating them risks applying different feature names, imputation, encoding, or funding transformation at inference time.

### What to use for the rubric

Use Render Web Service and submit its public URL with the repository. A public URL demonstrates cross-platform access from phones without requiring the evaluator to be on your Wi-Fi. Use the free tier for the demonstration; use a paid always-on instance if the app must respond immediately and continuously. The model is part of the web service deployment, not a second website.

For a Python hosting service, connect this repository and configure:

| Setting | Value |
| --- | --- |
| Python | 3.12 |
| Working directory | Repository root |
| Install command | `pip install -r webapp/requirements.txt` |
| Start command | `python -m webapp.app` |
| Environment | `HOST=0.0.0.0`; use the platform's `PORT` |
| Health path | `/health` |

The production server reads `HOST` and `PORT` from the environment. Put HTTPS at the hosting platform's reverse proxy. For a shared public deployment, configure request-rate limits at that proxy and budget enough memory for Python, scikit-learn, and the loaded forest. The app has no database or user accounts.

Alternatively, Docker provides a platform-independent deployment package. Run from the repository root:

```sh
docker build -f webapp/Dockerfile -t canopy-startup .
docker run --rm -p 8000:8000 canopy-startup
```

The image installs only the web app dependencies, copies the saved model, and runs as an unprivileged user. The Dockerfile-specific ignore file limits the build context to the webapp. Docker requires a working Docker installation; the container commands must be verified on the deployment host.

## Inputs and behavior

- Startup name is an optional display label and never enters the model.
- The seven predictors are total funding in USD, funding rounds, founding year, first funding year, last funding year, country, and one industry category.
- Unknown fields may be blank. At least one model input is required, and sparse inputs trigger an explanation.
- Funding must be non-negative; rounds must be positive whole numbers; dates must be valid whole years through the current year.
- Reversed funding dates and founding after last funding are flagged. For pre-incorporation funding, the user can leave the uncertain founding year blank, consistent with the conservative training cleaning.
- Dates newer than the 2015 training snapshot and funding outside training ranges display warnings.
- The example menu offers Northstar Labs (well-funded software), Harbor Cart (early-stage e-commerce), Seedling Health (healthcare with sparse details), and a clear-form option. These are fictional profiles and do not guarantee any outcome. Reset clears inputs, results, and errors. Editing inputs clears any stale result, and outstanding requests are cancelled.
- Predictions are processed in memory. This app does not save submitted names or features, use browser storage, or call a third-party prediction service. Hosting providers may keep their own HTTP access logs.

## Model and interpretation

Random Forest is finalized: 100 trees, `max_depth=None`, `min_samples_leaf=5`, and seed 42. The complete saved pipeline includes the funding `log1p` transformation, training-median imputation, missing indicators, rare-category one-hot encoding, and classifier. The frontend submits raw funding; it never applies the log transformation itself.

The artifact is trained on the same 9,429 training companies as the notebook, preserving its untouched 2,358-company evaluation split. Its test accuracy is **74.26%**, macro F1 **0.7413**, and ROC AUC **0.8266**. Every test prediction and score was checked against `results/random_forest_predictions.csv`.

The displayed classes are **Success pattern (acquired)** and **Failure pattern (closed)**. The two scores come from `predict_proba`, and are explicitly labeled as uncalibrated model scores. Acquisition is the dataset's success proxy. Operating and IPO companies were excluded, and the historical snapshot ends in 2015. The model does not estimate future survival duration or establish reliable performance on present-day startups. Historical funding may also include information recorded after the outcome.

To rebuild the artifact after intentional training changes:

```powershell
.\.venv\Scripts\python.exe -m webapp.train_model
```

This reads the existing cleaned CSV and verifies predictions against the existing notebook export before saving. If the training recipe changes intentionally, regenerate the notebook and its reference predictions first. Restart the server after rebuilding; artifacts are loaded once at startup. Load only the trusted model artifact generated by this project, and use the pinned scikit-learn version.

## Accessibility and responsive design

The page uses a two-column desktop layout and a stacked mobile layout. Narrow screens also stack the input fields. It provides native labeled inputs, number-friendly mobile keyboards, visible keyboard focus, a skip link, field-specific validation, live result announcements, touch-sized controls, and reduced-motion support. Results use text labels as well as color. Fonts and assets are local, with no CDN dependency.

## Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s webapp/tests -v
```

Tests cover the real model's holdout parity, input validation, missing values, chronology, malformed and oversized requests, score totals, current-date warnings, and the page/health endpoints. Browser checks and their verified viewport sizes are recorded in `QA.md`.

## Folder layout

```text
webapp/
  app.py                  Flask routes, validation, Waitress entry point
  model.py                Artifact loading and prediction
  train_model.py          Reproducible model export
  artifacts/              Saved pipeline and evaluation metadata
  templates/index.html    Accessible page structure
  static/                 CSS, JavaScript, and favicon
  tests/test_app.py        Inference and API tests
  requirements.txt        Pinned runtime dependencies
  Dockerfile              Container deployment
  Dockerfile.dockerignore Limited build context
  QA.md                   Verification record
```

Reference: Flask's official [Waitress deployment documentation](https://flask.palletsprojects.com/en/stable/deploying/waitress/).

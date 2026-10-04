# Verification record

## Model and API

Command: `python -m unittest discover -s webapp/tests -v`

All six test groups passed:

1. Model artifact matches all 2,358 notebook holdout predictions and probability scores, with score tolerance 1e-12.
2. Page and health endpoints respond correctly.
3. Successful predictions have valid scores; startup names do not affect inference; responses are not cached.
4. Unknown inputs work, and sparse-input/current-date warnings appear.
5. Negative, non-finite, fractional, out-of-range, Boolean, and malformed numeric inputs are rejected.
6. Invalid categories, reversed dates, all-empty input, malformed JSON, and oversized JSON requests are rejected.

## Browser checks

Checked in the Codex in-app browser on Windows. Widths below are measured CSS viewport widths, accounting for the browser's display scale.

| Viewport | Result |
| --- | --- |
| 320px phone | No horizontal overflow; form, controls, and result fit the viewport |
| 390px phone | Stacked layout; example submission completes and moves to the result |
| 768px tablet | Form, inputs, buttons, and result fit the viewport without horizontal overflow |
| 1280px desktop | Two-column interface and working prediction result |

Verified through the browser:

- Example fills all seven predictors and shows **Success pattern**, with acquired score 68.0% and closed score 32.0%.
- A lower-funding profile shows **Failure pattern**, with acquired score 21.6% and closed score 78.4%.
- Submitting an empty form produces a clear validation message.
- Negative funding produces a field-specific error.
- Using a 2026 funding date shows the warning about the 2015 training snapshot.
- Reset and input edits clear stale predictions.
- No JavaScript error/warning messages were recorded during the checked flows.
- Temporary viewport overrides were reset after testing.

A full-page [preview screenshot](screenshots/preview.jpg) and a [result screenshot](screenshots/result.jpg) are included. Native labels, keyboard focus styling, live result regions, reduced-motion CSS, and touch-sized controls are implemented. Physical iPhone/Android hardware, Safari, screen-reader behavior, and real phone-to-computer network connectivity were not tested.

## Deployment status

The Flask application was started locally with Waitress at `http://127.0.0.1:8000`. Public internet hosting has not been provisioned. The README covers a LAN phone demonstration, a Python hosting service, and Docker deployment.

Docker client software is present, but the Docker engine was unavailable, so the container build was not verified. The Dockerfile and its deployment instructions are provided for verification on a running Docker host.

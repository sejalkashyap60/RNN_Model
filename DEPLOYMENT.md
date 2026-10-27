# Streamlit Community Cloud deployment

1. Push this folder's source to a GitHub repository.
2. Confirm `requirements.txt` is at the repository root (or set the app root to this folder).
3. Open Streamlit Community Cloud and connect GitHub if prompted.
4. Create an app using the repository, branch, and `app.py` entry point.
5. Deploy, then verify training, architecture views, predictions, sequence exploration, experiment checkpoints, and held-out evaluation in the public app.

The app requires no API keys or dataset downloads. Checkpoints and experiment summaries are written to runtime storage and may be lost after restarts or redeploys. Use external durable storage only if persistent checkpoints are required. Check startup logs if dependency resolution or import fails. A component selector provides diagram inspection because Streamlit's Graphviz view does not support native click-to-inspect callbacks.

Local command: `python -m streamlit run app.py`.

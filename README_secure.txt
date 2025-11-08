# Secure-Ready Variant

This variant avoids hardcoding secrets and offers address privacy controls.

## Secrets

- **Streamlit Cloud**: In the app's Settings → *Secrets*, add:
  - `CENSUS_API_KEY = your_key_here`
  - (optional) `APP_PASSWORD = a_strong_password`

- **Local**: Copy `.env.example` to `.env` and set values, or export env vars:
  ```bash
  export CENSUS_API_KEY=your_key
  export APP_PASSWORD=somepassword
  ```

## Run

```bash
pip install -r requirements.txt
streamlit run app_secure.py
```

## Privacy Controls

- **Jitter coordinates**: Adds ±150m noise to obscure exact locations.
- **Neighborhood-only**: Replaces exact points with ZIP-level centroids.
- **Password gate**: If `APP_PASSWORD` is set, users must enter it in the sidebar.
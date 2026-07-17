# 💳 Credit Approval Prediction App

An interactive **Streamlit** web application that predicts a customer's **credit approval risk level** using a pre-trained **XGBoost classifier** (`xgb_classifier.pkl`). Users enter customer financial and credit-bureau details in a sidebar form, and the app returns a probability distribution across four risk bands in real time.

The app is containerized with **Docker** and deployed to **AWS EKS (Kubernetes)** via a **GitHub Actions** CI/CD pipeline.

---

## 🚀 Features

- **Interactive Web UI** — built with Streamlit for a fast, no-frontend-code experience.
- **Dynamic Inputs** — sidebar form is generated automatically from `model.feature_names_in_`, so the UI always matches the model's actual feature contract.
- **Real-Time Prediction** — the trained XGBoost classifier scores the customer instantly.
- **Visual Output** — a probability bar chart plus a four-metric breakdown.
- **Input Transparency** — every submitted value is echoed back with a human-readable label.
- **One-click cloud deployment** — push to `main` and GitHub Actions builds, pushes, and deploys automatically.

---

## 🧠 Model Details

| | |
|---|---|
| **Model Type** | XGBoost multi-class classifier (`xgboost.sklearn.XGBClassifier`) |
| **Serialized File** | `xgb_classifier.pkl` |
| **Input Features** | 54 numeric/categorical fields (income, employment tenure, loan-holding flags, trade-line counts, delinquency history, credit inquiries, demographics) |
| **Output** | Probability distribution across 4 risk classes |

**Risk classes:**
- **P1** — Low Risk
- **P2** — Medium Risk
- **P3** — High Risk
- **P4** — Near to Default

> ⚠️ **Known inconsistency:** the bar chart currently labels these P1–P4 as risk levels, while the metric cards below label the same values as "Approval" levels (e.g. "High Approval"). These are two different framings of the same 4 classes — pick one convention when extending the app to avoid confusing end users.

---

## 📁 Project Structure

```
.
├── app.py                        # Streamlit app: UI + inference (single-file application)
├── xgb_classifier.pkl            # Trained XGBoost model (required at runtime)
├── requirements.txt              # Pinned Python dependencies
├── Dockerfile                    # Container build definition
├── deployment.yaml               # Kubernetes Deployment (2 replicas)
├── service.yaml                  # Kubernetes Service (LoadBalancer, port 80 -> 8501)
├── devcontainer.json             # Codespaces / VS Code Dev Container config
├── .github/workflows/deploy.yml  # CI/CD: build -> push to ECR -> deploy to EKS
└── README.md                     # This file
```

---

## 🏗️ Architecture

```
Browser ──HTTP/WebSocket──▶ K8s Service (LoadBalancer, :80)
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                             ▼
            Pod: credit-app #1              Pod: credit-app #2
            Streamlit + app.py               Streamlit + app.py
            (model loaded into memory)        (model loaded into memory)

Release pipeline: git push (main) → GitHub Actions → Docker build →
push image to AWS ECR → kubectl apply → EKS rolling update
```

This is a **stateless, single-service monolith** — one container does both UI rendering and inference. There is no separate API layer, database, or message queue.

---

## 🧩 Column Mapping

`app.py` includes a `COLUMN_NAMES` dictionary that maps raw model feature names to human-readable UI labels. It affects display only — it has no effect on predictions.

| Model Column | UI Display Name |
|---------------|-----------------|
| `NETMONTHLYINCOME` | Net Monthly Income |
| `Time_With_Curr_Empr` | Employment Tenure |
| `CC_Flag` | Credit Card Holder |
| `EDUCATION` | Education Level |
| `num_deliq_6_12mts` | Delinquencies in 6–12 Months |

---

## ⚙️ Installation & Setup (Local)

1. **Clone the repository**
   ```bash
   git clone https://github.com/UjjwalKaushik0609/credit-risk-app
   cd credit-risk-app
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure the model file is present**
   `xgb_classifier.pkl` must be in the project root (already included in this repo).
4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Opens automatically at `http://localhost:8501`.

---

## 🐳 Docker

```bash
docker build -t credit-app .
docker run -p 8501:8501 credit-app
```

---

## ☸️ Kubernetes / AWS EKS Deployment

`deployment.yaml` and `service.yaml` describe the desired runtime state:

- **Deployment** (`credit-app`) — runs 2 replica pods of the container image.
- **Service** (`credit-service`) — a `LoadBalancer` service that provisions a cloud load balancer and forwards public port `80` to each pod's port `8501`.

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### CI/CD

`.github/workflows/deploy.yml` runs automatically on every push to `main`:

1. Checks out the repo.
2. Authenticates to AWS using GitHub Actions secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ACCOUNT_ID`).
3. Logs in to Amazon ECR.
4. Builds the Docker image and tags it for ECR.
5. Pushes the image to ECR.
6. Updates `kubeconfig` for the `credit-cluster` EKS cluster (region `ap-south-1`).
7. Applies `deployment.yaml` and `service.yaml`, triggering a rolling update on EKS.

---

## 🖥️ How It Works

1. Open the app in your browser.
2. Use the **sidebar** to enter customer details (income, education, loan flags, bureau history, etc.).
3. Click **"Predict Credit Approval"**.
4. View:
   - A **bar chart** of the 4 class probabilities.
   - **Metric cards** with each risk level's probability.
   - A **table** of everything you entered, for transparency.

---

## 🧰 Dependencies

Key libraries (see `requirements.txt` for the full pinned list):

- `streamlit`
- `pandas`
- `numpy`
- `xgboost`
- `scikit-learn`

```bash
pip install streamlit pandas numpy xgboost scikit-learn
```

---

## ⚠️ Known Limitations (current version)

- **No authentication/authorization** — anyone who can reach the endpoint can use the app.
- **No HTTPS/TLS** configured at the Service/Ingress level.
- **No input bounds validation** on numeric fields (income, trade-line counts, delinquency counts, etc.) — only the binary/categorical fields are constrained via dropdowns.
- **No persistence** — predictions are not logged or stored anywhere; there is no database.
- **No REST API** — the app is UI-only; nothing can call it programmatically today.
- **Model is loaded on every rerun** rather than cached (`@st.cache_resource` would fix this).
- **No health/readiness probes** defined in `deployment.yaml`.

---

## 💡 Future Enhancements

- Add input validation and bounds-checking on all numeric fields.
- Cache the model load with `@st.cache_resource` for better performance.
- Add a `POST /predict` REST API (e.g. via FastAPI) so other systems can call the model programmatically.
- Add authentication (SSO) and role-based access control.
- Persist predictions (inputs, outputs, model version) to a database for audit trail and drift monitoring.
- Add model explainability using SHAP.
- Add Kubernetes readiness/liveness probes and resource requests/limits.
- Export results as PDF or CSV.
- Move the model artifact to a model registry (e.g. MLflow) instead of a committed pickle file.

---

## 🧑‍💻 Author

**Developed by:** Ujjwal Kaushik
**Role:** Data Scientist / Machine Learning Engineer
**Contact:** ujjwalkaushik0609@gmail.com
**LinkedIn:** [linkedin.com/in/ujjwal-kaushik-0b1489376](https://www.linkedin.com/in/ujjwal-kaushik-0b1489376)

---

**Enjoy predicting smarter credit decisions!**

import base64
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============ Background + global styling ============

def set_background(image_file: str):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        backdrop-filter: blur(12px);
        background: rgba(5, 0, 20, 0.78);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .stSidebar {{
        background: radial-gradient(circle at top left,
                    rgba(255, 75, 129, 0.28),
                    rgba(5, 0, 20, 0.96));
        backdrop-filter: blur(14px);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .stSidebar, .stSidebar * {{
        color: #ffffff !important;
    }}
    h1, h2, h3 {{
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #ffffff;
    }}
    p, li, label, span {{
        color: #f5f5ff;
    }}
    .stRadio > label {{
        font-weight: 600;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, #ff4b81, #ff9a62);
        border-radius: 999px;
        border: none;
        color: #ffffff;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
    }}
    .stButton>button:hover {{
        box-shadow: 0 0 18px rgba(255, 75, 129, 0.65);
        transform: translateY(-1px);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ============ Load models and feature names ============

@st.cache_resource
def load_models():
    dt  = joblib.load("dt_model_magic.pkl")
    ada = joblib.load("ada_model_magic.pkl")
    rf  = joblib.load("rf_model_magic.pkl")
    nb  = joblib.load("nb_model_magic.pkl")
    feature_names = joblib.load("magic_feature_names.pkl")

    models = {
        "Decision Tree": dt,
        "AdaBoost": ada,
        "Random Forest": rf,
        "Naive Bayes": nb,
    }
    best_model_name = "Random Forest"  # change if another model is best
    best_model = models[best_model_name]
    return models, best_model_name, best_model, feature_names


models, best_model_name, best_model, feature_names = load_models()
set_background("image.jpg")


# ============ Sidebar navigation ============

st.sidebar.title("MAGIC GAMMA CLASSIFIER")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Gamma Signal Classifier", "Dashboard", "Model Explanation"]
)


# ============ Helper: evaluate a model on a dataset ============

def evaluate_on_dataset(model, df, target_col="class"):
    X = df[feature_names]
    y = df[target_col]
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, pos_label="g"),
        "Recall": recall_score(y, y_pred, pos_label="g"),
        "F1": f1_score(y, y_pred, pos_label="g"),
    }


# ============ Overview page ============

if page == "Overview":
    st.title("MAGIC GAMMA VS HADRON CLASSIFIER")

    st.markdown("""
This application helps scientists distinguish between **gamma-ray events** (useful signal) 
and **hadron events** (background noise) detected by atmospheric Cherenkov telescopes 
such as the MAGIC telescope.  
""")

    st.markdown("""
- **Gamma rays**: High-energy photons from astrophysical sources (black holes, pulsars, explosions).  
- **Hadrons**: Cosmic-ray particles that create similar light flashes but are considered **noise**.  
- **Goal**: Automatically classify each event as **gamma (signal)** or **hadron (background)** to speed up analysis.
""")

    st.info(
        "Target users: astrophysicists, researchers, and technicians who want a simple tool "
        "to filter gamma events from large telescope datasets without dealing with ML code."
    )

    st.subheader("What this app offers")
    st.markdown("""
1. **Gamma Signal Classifier** – Enter or upload event data and get a prediction (gamma vs hadron).  
2. **Dashboard** – View class distribution and basic statistics of uploaded events.  
3. **Model Explanation** – See which features are most important for the best model.
""")


# ============ Gamma Signal Classifier page ============

elif page == "Gamma Signal Classifier":
    st.title("Gamma Signal Classifier")

    st.markdown(
        f"Best model from the assignment: **{best_model_name}** (trained on balanced MAGIC dataset)."
    )

    st.subheader("1. Upload CSV (optional)")
    st.markdown("""
Upload a MAGIC-like CSV file.  
If it has no header, the app will apply the standard 10 feature names plus `class`.  
If a `class` column is present, the app can also evaluate performance.
""")

    uploaded_file = st.file_uploader("Upload MAGIC-like CSV file", type=["csv"])

    df_uploaded = None
    if uploaded_file is not None:
        cols = [
            "fLength","fWidth","fSize","fConc","fConc1",
            "fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"
        ]
        df_uploaded = pd.read_csv(uploaded_file, header=None, names=cols)

        st.write("Preview of uploaded data:")
        st.dataframe(df_uploaded.head())

        missing = [c for c in feature_names if c not in df_uploaded.columns]
        if missing:
            st.error(f"Uploaded file is missing columns: {missing}")
        else:
            st.success("All required feature columns are present.")

            # Always allow prediction summary
            X_up = df_uploaded[feature_names]
            preds = best_model.predict(X_up)
            st.subheader("Predicted classes (count) for uploaded events")
            st.write(pd.Series(preds).value_counts())

            # Only compute metrics if 'class' is present
            if "class" in df_uploaded.columns:
                metrics = evaluate_on_dataset(best_model, df_uploaded)
                st.markdown("**Performance of best model on uploaded data:**")
                st.write(pd.DataFrame([metrics]).style.format("{:.4f}"))

    st.subheader("2. Manual event input")
    st.markdown("Enter feature values for a single event and click Predict:")

    cols_ui = st.columns(2)
    user_inputs = []
    for i, feat in enumerate(feature_names):
        with cols_ui[i % 2]:
            val = st.number_input(feat, value=0.0, format="%.4f")
            user_inputs.append(val)

    if st.button("Predict event class"):
        x_new = np.array(user_inputs).reshape(1, -1)
        pred = best_model.predict(x_new)[0]

        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(x_new)[0]
            prob_gamma = proba[list(best_model.classes_).index("g")]
            prob_hadron = proba[list(best_model.classes_).index("h")]
        else:
            prob_gamma = prob_hadron = None

        if pred == "g":
            st.success("Predicted class: **Gamma (signal)**")
        else:
            st.warning("Predicted class: **Hadron (background)**")

        if prob_gamma is not None:
            st.markdown(
                f"- Probability Gamma: **{prob_gamma:.3f}**  \n"
                f"- Probability Hadron: **{prob_hadron:.3f}**"
            )

        st.caption(
            "Gamma events are likely real astrophysical signals, while hadron events are background noise."
        )


# ============ Dashboard page ============

elif page == "Dashboard":
    st.title("Dataset Dashboard")

    st.markdown("Upload a labeled CSV file to see class distribution and basic statistics.")

    uploaded_file = st.file_uploader("Upload CSV with MAGIC columns (with or without header)", type=["csv"])

    if uploaded_file is not None:
        cols = [
            "fLength","fWidth","fSize","fConc","fConc1",
            "fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"
        ]
        df_dash = pd.read_csv(uploaded_file, header=None, names=cols)

        if "class" not in df_dash.columns:
            st.error("The uploaded file must contain a 'class' column with 'g' and 'h'.")
        else:
            st.subheader("Class distribution")
            class_counts = df_dash["class"].value_counts()
            st.write(class_counts)

            col1, col2 = st.columns(2)
            with col1:
                fig = class_counts.plot.pie(
                    autopct="%1.1f%%", ylabel="", title="Gamma vs Hadron"
                ).get_figure()
                st.pyplot(fig)
            with col2:
                st.write("Basic statistics for features:")
                st.write(df_dash[feature_names].describe())

            if set(df_dash["class"].unique()) <= {"g", "h"}:
                metrics = evaluate_on_dataset(best_model, df_dash)
                st.subheader("Best model performance on this dataset")
                st.write(pd.DataFrame([metrics]).style.format("{:.4f}"))
    else:
        st.info("No file uploaded yet. Use the uploader above to explore a dataset.")


# ============ Model Explanation page ============

elif page == "Model Explanation":
    st.title("Model Explanation")

    st.markdown(
        f"This page explains which features are most important for the **{best_model_name}** model."
    )

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.subheader("Feature importances")
        st.bar_chart(imp_df.set_index("Feature"))

        st.markdown("Top features (most influence on prediction):")
        st.write(imp_df.head())
    else:
        st.info("The selected best model does not provide feature_importances_.")

    st.caption(
        "Higher importance means the model relies more on that feature when deciding "
        "if an event is gamma or hadron."
    )

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit page info
st.set_page_config(
    page_title="Forward Insights – Campaign Planner",
    layout="wide"
)

# income / education mappings (Pew survey style)
INCOME_MAP = {
    "Less than $10,000": 1,
    "$10,000 to under $20,000": 2,
    "$20,000 to under $30,000": 3,
    "$30,000 to under $40,000": 4,
    "$40,000 to under $50,000": 5,
    "$50,000 to under $75,000": 6,
    "$75,000 to under $100,000": 7,
    "$100,000 to under $150,000": 8,
    "$150,000 or more": 9
}

EDU_MAP = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Two-year associate degree": 5,
    "Four-year college/university degree": 6,
    "Some postgraduate or professional schooling": 7,
    "Postgraduate or professional degree": 8
}


def as_binary(x):
    return np.where(x == 1, 1, 0)


@st.cache_data
def load_data(path="social_media_usage.csv"):
    df_raw = pd.read_csv(path)

    # clean the stuff 
    df = pd.DataFrame({
        "is_user": as_binary(df_raw["web1h"]),
        "income": np.where(df_raw["income"] <= 9, df_raw["income"], np.nan),
        "education": np.where(df_raw["educ2"] <= 8, df_raw["educ2"], np.nan),
        "parent": as_binary(df_raw["par"]),
        "married": as_binary(df_raw["marital"]),
        "female": np.where(df_raw["sex"] == 2, 1, 0),
        "age": np.where(df_raw["age"] <= 98, df_raw["age"], np.nan)
    })


    df = df.dropna()

    X_cols = ["income", "education", "parent", "married", "female", "age"]
    X = df[X_cols]
    y = df["is_user"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    results = {
        "acc": accuracy_score(y_test, preds),
        "prec": precision_score(y_test, preds),
        "rec": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "cm": confusion_matrix(y_test, preds),
        "baseline": df["is_user"].mean(),
        "cols": X_cols
    }

    return model, results


# visualization helpers
def show_drivers(model, cols):
    coef_df = (
        pd.DataFrame({"factor": cols, "impact": model.coef_[0]})
          .sort_values("impact", ascending=False)
    )

    return (
        alt.Chart(coef_df)
        .mark_bar()
        .encode(
            x="impact",
            y=alt.Y("factor", sort="-x", title=""),
            color=alt.condition(
                alt.datum.impact > 0,
                alt.value("#0077B5"),
                alt.value("#D32F2F")
            ),
            tooltip=["factor", "impact"]
        )
        .properties(title="Demographic Factors")
    )


def show_lifecycle(model, persona):
    # simulate across age 18 to 90
    ages = list(range(18, 91))
    sim = pd.DataFrame([persona] * len(ages))
    sim["age"] = ages

    sims = model.predict_proba(sim)[:, 1]

    df_curve = pd.DataFrame({"Age": ages, "Probability": sims})

    base = alt.Chart(df_curve)

    line = base.mark_line(strokeWidth=4, color="#0077B5").encode(
        x="Age",
        y=alt.Y("Probability", axis=alt.Axis(format=".0%"), title="Predicted Likelihood"),
        tooltip=["Age", alt.Tooltip("Probability", format=".1%")]
    )

    fifty = alt.Chart(pd.DataFrame({"p": [0.5]})).mark_rule(
        color="gray", strokeDash=[5, 5]
    ).encode(y="p")

    # highlight the persona's actual point
    p_current = model.predict_proba(pd.DataFrame([persona]))[0][1]
    point = (
        alt.Chart(pd.DataFrame({"Age": [persona["age"]], "Probability": [p_current]}))
        .mark_point(size=100, color="black")
        .encode(x="Age", y="Probability")
    )

    return (line + fifty + point).properties(title="Lifecycle Curve")


# APP
def main():
    try:
        model, results = load_data("social_media_usage.csv")
    except Exception:
        st.error("Could not find the dataset. Upload `social_media_usage.csv`.")
        return

    st.title("Forward Insights – Campaign Planner")
    st.write("LinkedIn usage simulator based on demographic traits.")

    # sidebar persona builder
    st.sidebar.header("Persona Setup")

    income_pick = st.sidebar.selectbox("Household Income", list(INCOME_MAP.keys()), index=7)
    edu_pick = st.sidebar.selectbox("Education Level", list(EDU_MAP.keys()), index=6)
    age_val = st.sidebar.slider("Age", 18, 98, 42)

    parent_flag = 1 if st.sidebar.radio("Parent", ["No", "Yes"]) == "Yes" else 0
    married_flag = 1 if st.sidebar.radio("Married", ["No", "Yes"], index=1) == "Yes" else 0
    female_flag = 1 if st.sidebar.radio("Gender", ["Male", "Female"], index=1) == "Female" else 0

    persona = {
        "income": INCOME_MAP[income_pick],
        "education": EDU_MAP[edu_pick],
        "parent": parent_flag,
        "married": married_flag,
        "female": female_flag,
        "age": age_val
    }

    # model prediction
    prob = model.predict_proba(pd.DataFrame([persona]))[0][1]

    st.subheader("Audience Strategy")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.metric(
            "Adoption Probability",
            f"{prob:.1%}",
            f"{prob - results['baseline']:.1%} vs avg"
        )

    with col_b:
        if prob >= 0.70:
            st.success(
                "Tier 1: Direct Conversion\n\n"
                "Strong likelihood of LinkedIn usage. Good for lead forms or demo pushes."
            )
        elif prob >= 0.50:
            st.info(
                "Tier 2: Nurture\n\n"
                "They’re on the platform but not heavy users. Better for educational content."
            )
        else:
            st.warning(
                "Tier 3: Suppress\n\n"
                "Not an efficient segment compared to cost. Possible skip group."
            )

    st.divider()

    st.subheader("Market View")

    left, right = st.columns(2)
    with left:
        st.write("Lifecycle View")
        st.altair_chart(show_lifecycle(model, persona), use_container_width=True)

    with right:
        st.write("Drivers")
        st.altair_chart(show_drivers(model, results["cols"]), use_container_width=True)

    st.divider()

    # model metrics
    with st.expander("Model Details"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{results['acc']:.2f}")
        m2.metric("Precision", f"{results['prec']:.2f}")
        m3.metric("Recall", f"{results['rec']:.2f}")
        m4.metric("F1 Score", f"{results['f1']:.2f}")

        st.write("Confusion Matrix")
        cm_df = pd.DataFrame(results["cm"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df)

    # footer
    st.markdown("---")
    st.caption("Created by **Jose Ochoa Tello** for Programming II – Final Project")

if __name__ == "__main__":
    main()

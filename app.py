"""
Titanic Survival Checker - Streamlit front-end for the SVC model trained in the notebook.
The model itself (svc_trained_model.pkl) is the one produced by the instructor's training code;
this file only handles the input form, label encoding and the way the answer is shown.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Checker", page_icon="⚓", layout="wide")

MODEL_FILE = Path(__file__).parent / "svc_trained_model.pkl"

# The same fixed category lists the notebook used to fit its LabelEncoders.
CHOICES = {
    "PClass": ["First", "Second", "Third"],
    "Gender": ["Male", "Female"],
    "Sibling": ["Zero", "One", "Two", "Three"],
    "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
}


@st.cache_resource
def load_everything():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    encoders = {name: LabelEncoder().fit(values) for name, values in CHOICES.items()}
    return model, encoders


def encode_passenger(passenger, encoders):
    """Turn the 4 words chosen in the form into the 4 numbers the model was trained on."""
    row = pd.DataFrame([passenger], columns=list(CHOICES))
    return pd.DataFrame({col: encoders[col].transform(row[col]) for col in CHOICES})


model, encoders = load_everything()

st.title("⚓ Titanic Survival Checker")
st.write("Describe a passenger in the sidebar and the trained Support Vector Classifier "
         "will say whether that passenger would have survived the disaster.")

with st.sidebar:
    st.header("Passenger details")
    passenger = {
        "PClass": st.radio("Ticket class", CHOICES["PClass"], index=2, horizontal=True),
        "Gender": st.radio("Gender", CHOICES["Gender"], horizontal=True),
        "Sibling": st.select_slider("Siblings / spouses on board", CHOICES["Sibling"]),
        "Embarked": st.selectbox("Port of embarkation", CHOICES["Embarked"]),
    }
    go = st.button("Check survival", type="primary", use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Passenger summary")
    st.table(pd.DataFrame(passenger, index=["Value"]).T)

with right:
    st.subheader("Model answer")
    if go:
        x = encode_passenger(passenger, encoders)
        survived = int(model.predict(x)[0]) == 1
        score = float(model.decision_function(x)[0])
        if survived:
            st.success("## Would SURVIVE")
        else:
            st.error("## Would NOT survive")
        st.metric("SVC decision score", f"{score:+.3f}",
                  help="Positive = survived side of the boundary, negative = not survived side.")
        with st.expander("Numbers sent to the model"):
            st.dataframe(x, hide_index=True)
    else:
        st.info("Choose the details on the left and press **Check survival**.")

st.caption("Teaching demo trained on a small 100-row Titanic sample. It reflects patterns in that sample, "
           "not a rigorous historical analysis.")

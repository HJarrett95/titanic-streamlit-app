###
# Code originally written by Harry Wang (https://github.com/harrywang/mini-ml/)
# It was modified for the purpose of teaching how to deploy a machine learning 
# model using Streamlit.
###

import streamlit as st
import pandas as pd
import joblib
from datetime import date
from PIL import Image

# load your machine learning model
tree_clf = joblib.load('model_dt.pickle')

### Streamnlit app code starts here

st.title('Titanic Survival Prediction')

with st.expander('Show sample of Titanic data'):
    df = pd.read_csv('titanic.csv') # adjust filename if needed
    st.dataframe(df.head(5))

st.sidebar.markdown('**Please provide passenger information**:')  # you can use markdown like this

# get inputs
with st.sidebar.form('inputs'):
    sex = st.selectbox('Sex', ['female', 'male'])
    age = (lambda d: date.today().year - d.year - ((date.today().month, date.today().day) < (d.month, d.day)))(
    st.date_input(
        "Date of birth",
        min_value=date(1900, 1, 1),
        max_value=date.today(),
        format="DD/MM/YYYY"))
    sib_sp = int(st.number_input('# of siblings / spouses aboard:', min_value=0, max_value=10, value=0))
    pclass = st.selectbox('Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
    fare = int(st.number_input('# of parents / children aboard:', min_value=0, max_value=10, value=0))

    submitted = st.form_submit_button('Predict')

# this is how to dynamically change text
prediction_state = st.markdown('calculating...')

### Now the inference part starts here

passenger = pd.DataFrame(
    {
        'Pclass': [pclass],
        'Sex': [sex], 
        'Age': [age],
        'SibSp': [sib_sp],
        'Fare': [fare]
    }
)

y_pred = tree_clf.predict(passenger)
proba = tree_clf.predict_proba(passenger)


# Displays survival probability as %
st.markdown(f"The survival probability: **{proba[0][1]*100:.2f}%**")

# Preparing the message to be displayed based on the prediction
if y_pred[0] == 0:
    msg = 'This passenger is predicted to be: **died**'
else:
    msg = 'This passenger is predicted to be: **survived**'

### Now add the prediction result to the Streamlit app

prediction_state.markdown(msg)
if y_pred[0] == 1:
    st.image(
        "Survived.jpg",
        use_container_width=True)
    st.markdown(
        "<span style='color: green; font-size: 0.8rem;'>Survived!</span>",
        unsafe_allow_html=True)
else:
    st.image(
        "Died.jpg",
        caption="Did not survive",
        use_container_width=True
    )

#Add explanation in an expandable/collapsible box
with st.expander("Explanation of Prediction"):
    st.markdown(
        """
        **Features used:**  
        - Ticket class (`Pclass`)  
        - Sex (`Sex`)  
        - Age (`Age`)  
        - Number of siblings/spouses aboard (`SibSp`)  
        - Number of parents/children aboard (`Fare`)  

        **Limitations:**  
        - This model is trained only on historical Titanic data; it may not generalize beyond it.  
        - It does not account for other factors such as cabin location, deck, or port of embarkation.  
        - Probabilities are approximate and not guarantees of survival.
        """,
        unsafe_allow_html=True
    )

import streamlit as st
import category_encoders as ce
import pandas as pd

class EncodingMethods:
    def __init__(self, dataset):
        self.dataset = dataset

    def BackwardDifferenceEncoder(self):
        st.write("### Backward Difference Encoder")
        variable = st.selectbox("Select categorical variable:", self.dataset.select_dtypes(include=['object']).columns.tolist())
        drop_invariant = st.checkbox("Drop invariant columns", value=False)
        return_df = st.checkbox("Return as DataFrame", value=True)
        handle_unknown = st.selectbox("Handle unknown values:", ["error", "return_nan", "value", "indicator"], index=2)
        handle_missing = st.selectbox("Handle missing values:", ["error", "return_nan", "value", "indicator"], index=2)
        
        if st.button("Apply Backward Difference Encoding", use_container_width=True, type='primary'):
            encoder = ce.BackwardDifferenceEncoder(cols=[variable], drop_invariant=drop_invariant, return_df=return_df, handle_unknown=handle_unknown, handle_missing=handle_missing)
            st.session_state["BackwardDifferenceEncoder"] = encoder
            st.success("Backward Difference Encoder added to session state!")
    
    def BaseNEncoder(self):
        st.write("### Base-N Encoder")
        variable = st.selectbox("Select categorical variable:", self.dataset.select_dtypes(include=['object']).columns.tolist())
        drop_invariant = st.checkbox("Drop invariant columns", value=False, key="baseN_drop")
        return_df = st.checkbox("Return as DataFrame", value=True, key="baseN_df")
        base = st.slider("Select Base (N):", min_value=1, max_value=10, value=2)
        handle_unknown = st.selectbox("Handle unknown values:", ["error", "return_nan", "value", "indicator"], index=2, key="baseN_unknown")
        handle_missing = st.selectbox("Handle missing values:", ["error", "return_nan", "value", "indicator"], index=2, key="baseN_missing")
        
        if st.button("Apply Base-N Encoding", use_container_width=True, type='primary'):
            encoder = ce.BaseNEncoder(cols=[variable], drop_invariant=drop_invariant, return_df=return_df, base=base, handle_unknown=handle_unknown, handle_missing=handle_missing)
            st.session_state["BaseNEncoder"] = encoder
            st.success("Base-N Encoder added to session state!")

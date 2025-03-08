class Encoders:
    def __init__(self, dataset):
        self.dataset = dataset

    def common_parameters(self):
        # Streamlit widgets for user inputs
        verbose = st.slider('Verbose', 0, 3, 0)  # Slider for verbosity level
        cols = st.multiselect('Select Columns to Encode', options=self.dataset.columns.tolist())  # Select columns
        drop_invariant = st.checkbox('Drop Invariant Columns', value=False)  # Checkbox for drop invariant  # Base input
        
        return verbose, cols, drop_invariant

    def backward_difference_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.backward_difference.BackwardDifferenceEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
        )
        return encoder.fit_transform(self.dataset)

    def base_n_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.base_n.BaseNEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def binary_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        base = st.number_input('Base (Higher values for non-linear models)', min_value=2, max_value=10, value=2)
        encoder = ce.binary.BinaryEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            base=base,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def catboost_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.cat_boost.CatBoostEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def count_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.count.CountEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def glmm_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.glmm.GLMMEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def gray_encoding(self):
        verbose, cols, drop_invariant, base = self.common_parameters()
        base = st.number_input('Base (Higher values for non-linear models)', min_value=2, max_value=10, value=2)
        encoder = ce.gray.GrayEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            base=base,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def hashing_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.hashing.HashingEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            hash_method='md5',
            process_creation_method='fork'
        )
        return encoder.fit_transform(self.dataset)

    def helmert_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.helmert.HelmertEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def james_stein_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.james_stein.JamesSteinEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def leave_one_out_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        sigma=st.number_input("adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched). Sigma gives the standard deviation (spread or “width”) of the normal distribution. The optimal value is commonly between 0.05 and 0.6. The default is to not add noise, but that leads to significantly suboptimal results.",0.5)
        encoder = ce.leave_one_out.LeaveOneOutEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
            sigma=sigma
        )
        return encoder.fit_transform(self.dataset)

    def m_estimate_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        base = st.number_input("this is the “m” in the m-probability estimate. Higher value of m results into stronger shrinking. M is non-negative.",1.0)
        sigma = st.number_input('standard deviation (spread or “width”) of the normal distribution.',0.05)
        randomized=st.checkbox("adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).")
        encoder = ce.m_estimate.MEstimateEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
            m=base,
            sigma=sigma,
            randomized=randomized
        )
        return encoder.fit_transform(self.dataset)

    def one_hot_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.one_hot.OneHotEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
            use_cat_names=True
        )
        return encoder.fit_transform(self.dataset)

    def ordinal_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.ordinal.OrdinalEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def polynomial_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.polynomial.PolynomialEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def quantile_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        base = st.number_input('float indicating statistical quantile. ´0.5´ for median.')
        m=st.number_input('this is the “m” in the m-probability estimate. Higher value of m results into stronger shrinking. M is non-negative. 0 for no smoothing.')
        encoder = ce.quantile_encoder.QuantileEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_missing='value',
            handle_unknown='value',
            quantile=base,
            m=m
        )
        return encoder.fit_transform(self.dataset)

    def rankhot_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.rankhot.RankHotEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_missing='value',
            handle_unknown='value'
        )
        return encoder.fit_transform(self.dataset)

    def sum_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        encoder = ce.sum_coding.SumEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value'
        )
        return encoder.fit_transform(self.dataset)

    def summary_encoding(self):
        verbose, cols, drop_invariant= self.common_parameters()
        quantiles=st.text_input("comma separated - list of floats indicating the statistical quantiles. Each element represent a column")
        base = st.number_input('this is the “m” in the m-probability estimate. Higher value of m results into stronger shrinking. M is non-negative. 0 for no smoothing.',1.0)
        encoder = ce.quantile_encoder.SummaryEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_missing='value',
            handle_unknown='value',
            quantiles=[eval(x) for x in quantiles.split(',') if quantiles else (0.25,0.75)]
            m=base
        )
        return encoder.fit_transform(self.dataset)

    def target_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        min_samples_leaf=int(st.number_input("For regularization the weighted average between category mean and global mean is taken. The weight is an S-shaped curve between 0 and 1 with the number of samples for a category on the x-axis. The curve reaches 0.5 at min_samples_leaf. (parameter k in the original paper)",20))
        smoothing=st.number_input("smoothing effect to balance categorical average vs prior. Higher value means stronger regularization. The value must be strictly bigger than 0. Higher values mean a flatter S-curve (see min_samples_leaf).",10.0)
        encoder = ce.target_encoder.TargetEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_missing='value',
            handle_unknown='value',
            smoothing=smoothing,
            min_samples_leaf=min_samples_leaf
        )
        return encoder.fit_transform(self.dataset)

    def woe_encoding(self):
        verbose, cols, drop_invariant = self.common_parameters()
        sigma = st.number_input('standard deviation (spread or “width”) of the normal distribution.',0.05)
        randomized=st.checkbox("adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).")
        regularization=st.number_input('the purpose of regularization is mostly to prevent division by zero. When regularization is 0, you may encounter division by zero.',1.0)
        encoder = ce.woe.WOEEncoder(
            verbose=verbose,
            cols=cols if cols else None,
            drop_invariant=drop_invariant,
            return_df=True,
            handle_unknown='value',
            handle_missing='value',
            sigma=sigma,
            randomized=randomized,
            regularization=regularization
        )
        return encoder.fit_transform(self.dataset)

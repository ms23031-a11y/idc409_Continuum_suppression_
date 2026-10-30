"""Streamlit demo: score uploaded Belle II feature rows with the SAVED best signal-vs-continuum pipeline and its frozen operating threshold.
Run:  streamlit run app.py
Labels follow the instructor-provided mapping (relayed by the student): type +1 = signal, type 0/-1 = continuum. Rows of other decays (type 2-5) are out of scope."""
import pandas as pd, streamlit as st
from src import predict
from src.paths import MET, MOD

st.set_page_config(page_title="Continuum suppression demo", layout="wide")
st.title("Belle II continuum suppression - demo")
st.info("Signal-vs-continuum classifier trained on rows with `type` in {+1 (signal), 0 / -1 (continuum)}; mapping provided by the instructor via the student. Events of other decays (`type` 2-5) are out of scope: they were excluded from training and the model gives them no meaningful class.")
bundle = predict.load_bundle(MOD / "best_model.joblib"); neg, pos = predict.class_names(bundle)
st.caption(f"Model: {bundle['model_name']} ({bundle['feature_set']} features) | {len(bundle['features'])} features | threshold {bundle['threshold']:.3f} (chosen on validation data: max continuum rejection subject to the signal-efficiency floor)")
_r = pd.read_csv(MET / "results_full.csv").set_index("key").loc[f"{bundle['model_name']}|{bundle['feature_set']}"]       # held-out numbers of THIS saved model
st.warning(f"Held-out test performance of this saved model (from the saved evaluation): ROC-AUC {_r.test_auc:.3f} [{_r.auc_ci_lo:.3f}, {_r.auc_ci_hi:.3f}]; signal efficiency {_r.op_signal_efficiency:.1%}; continuum rejection {_r.op_continuum_rejection:.1%} "
           f"(only {int(_r.n_continuum_test)} continuum test events). Scores below are indicative only.")
up = st.file_uploader("CSV with the required feature columns (extra columns such as `type` are ignored)", type="csv")
if up:
    df = pd.read_csv(up)
    try:
        out = predict.predict_df(df, bundle)
    except ValueError as e:
        st.error(str(e)); st.stop()
    res = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    counts = res["predicted_class"].value_counts()
    st.success(f"Scored {len(res)} rows with the saved pipeline. Predicted class counts: " + "; ".join(f"{k}: {v}" for k, v in counts.items()))
    st.dataframe(res[["signal_score", "predicted_class"] + [c for c in df.columns if c in ("type",)]].head(200))
    st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv")
with st.expander("Required feature columns"): st.write(bundle["features"])

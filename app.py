"""Titanic Passenger Insights — Programming for Data Science project."""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
DATA_PATH = Path(__file__).with_name("titanic.csv")
FIELDS = ["survived", "pclass", "sex", "age", "fare", "embarked"]
AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
AGE_BINS = [0, 13, 19, 36, 61, float("inf")]
CLASS_LABELS = {1: "First class", 2: "Second class", 3: "Third class"}
CLASS_COLORS = ["#18344A", "#168B83", "#92BEB7"]
SEX_COLORS = ["#168B83", "#365D7E"]
OUTCOME_LABELS = ["Did not survive", "Survived"]
OUTCOME_COLORS = ["#C27C72", "#168B83"]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Prefer the bundled CSV so the app also works without internet access."""
    raw = pd.read_csv(DATA_PATH if DATA_PATH.is_file() else DATA_URL)
    missing_columns = set(FIELDS) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(missing_columns)}")
    return raw


def prepare_data(raw: pd.DataFrame):
    """Preserve raw records and create cleaned and transformed views."""
    # .loc selects all passengers and the six fields required by the worksheet.
    cleaned = raw.loc[:, FIELDS].copy()
    before = cleaned[["age", "embarked"]].isna().sum()
    age_median = cleaned["age"].median()
    embarked_mode = cleaned["embarked"].mode().iloc[0]
    cleaned["age"] = cleaned["age"].fillna(age_median)
    cleaned["embarked"] = cleaned["embarked"].fillna(embarked_mode)
    after = cleaned[["age", "embarked"]].isna().sum()

    transformed = cleaned.copy()
    for column in ["age", "fare"]:
        minimum = cleaned[column].min()
        maximum = cleaned[column].max()
        # The guard also handles a constant column without dividing by zero.
        transformed[f"{column}_norm"] = (
            (cleaned[column] - minimum) / (maximum - minimum)
            if maximum > minimum else 0.0
        )

    # Completed-year age bands: [0,13), [13,19), [19,36), [36,61), [61,infinity).
    transformed["age_group"] = pd.cut(
        cleaned["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    dummy_data = pd.get_dummies(
        cleaned[["sex", "embarked"]], columns=["sex", "embarked"], dtype=int
    )
    # Retain the readable categories for filters and keep dummy columns separate.
    transformed = transformed.join(dummy_data)
    details = {
        "age_median": age_median,
        "embarked_mode": embarked_mode,
        "missing": pd.DataFrame({"Before": before, "After": after}),
        "dummy_columns": dummy_data.columns.tolist(),
    }
    return cleaned, transformed, details


def filter_data(data, classes, sexes, age_groups):
    mask = (
        data["pclass"].isin(classes)
        & data["sex"].isin(sexes)
        & data["age_group"].isin(age_groups)
    )
    return data.loc[mask].copy()


def group_rates(data, group):
    return (
        data.groupby(group, observed=True)["survived"]
        .agg(passengers="size", survivors="sum", survival_rate="mean")
        .reset_index()
    )


def descriptive_finding(data):
    """Derive a finding from the current selection, never from a fixed sentence."""
    if data.empty:
        return "No passengers match the current filters. Broaden the selection to explore a pattern."
    for field, description in [("sex", "sex groups"), ("pclass", "passenger classes"), ("age_group", "age groups")]:
        rates = group_rates(data, field)
        if len(rates) < 2:
            continue
        rates = rates.sort_values("survival_rate")
        low, high = rates.iloc[0], rates.iloc[-1]
        def label(value):
            return CLASS_LABELS[int(value)] if field == "pclass" else str(value).capitalize()
        return (
            f"Among the selected {description}, {label(high[field])} passengers had a survival rate of "
            f"{high['survival_rate']:.1%} (n={int(high['passengers'])}), compared with "
            f"{low['survival_rate']:.1%} for {label(low[field])} passengers "
            f"(n={int(low['passengers'])}). This is an observed association in this dataset."
        )
    return (
        f"In this selected group, {int(data['survived'].sum())} of {len(data)} passengers "
        f"survived ({data['survived'].mean():.1%}). Broaden the filters to compare groups."
    )


def style_chart(chart):
    return (
        chart.properties(height=285)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#E9EEF1", labelColor="#536776", titleColor="#304B5D")
        .configure_legend(title=None, labelColor="#536776")
    )


def survival_counts_chart(data):
    counts = data["survived"].value_counts().reindex([0, 1], fill_value=0)
    table = pd.DataFrame({"Outcome": OUTCOME_LABELS, "Passengers": counts.to_numpy()})
    chart = alt.Chart(table).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=80).encode(
        x=alt.X("Outcome:N", sort=OUTCOME_LABELS, title="Survival outcome", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Passengers:Q", title="Number of passengers", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("Outcome:N", scale=alt.Scale(domain=OUTCOME_LABELS, range=OUTCOME_COLORS), legend=None),
        tooltip=["Outcome:N", "Passengers:Q"],
    )
    return style_chart(chart)


def rate_chart(data, group):
    table = group_rates(data, group)
    if group == "pclass":
        table["Group"] = table[group].map(CLASS_LABELS)
        order, colors, title = list(CLASS_LABELS.values()), CLASS_COLORS, "Passenger class"
    else:
        table["Group"] = table[group].str.capitalize()
        order, colors, title = ["Female", "Male"], SEX_COLORS, "Sex"
    chart = alt.Chart(table).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=65).encode(
        x=alt.X("Group:N", sort=order, title=title, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("survival_rate:Q", title="Survival rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        color=alt.Color("Group:N", scale=alt.Scale(domain=order, range=colors), legend=None),
        tooltip=[alt.Tooltip("Group:N", title=title), alt.Tooltip("survival_rate:Q", title="Survival rate", format=".1%"),
                 alt.Tooltip("passengers:Q", title="Passengers"), alt.Tooltip("survivors:Q", title="Survivors")],
    )
    return style_chart(chart)


def age_histogram(data):
    chart = alt.Chart(data).mark_bar(color="#168B83", binSpacing=2).encode(
        x=alt.X("age:Q", bin=alt.Bin(step=5, extent=[0, 85]), title="Age (years)", scale=alt.Scale(domain=[0, 85])),
        y=alt.Y("count():Q", title="Number of passengers", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("age:Q", bin=alt.Bin(step=5, extent=[0, 85]), title="Age range"),
                 alt.Tooltip("count():Q", title="Passengers")],
    )
    return style_chart(chart)


def fare_chart(data):
    table = data[["pclass", "fare"]].copy()
    table["Passenger class"] = table["pclass"].map(CLASS_LABELS)
    chart = alt.Chart(table).mark_boxplot(size=45, extent=1.5).encode(
        x=alt.X("Passenger class:N", sort=list(CLASS_LABELS.values()), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("fare:Q", title="Fare (historical pounds)"),
        color=alt.Color("Passenger class:N", scale=alt.Scale(domain=list(CLASS_LABELS.values()), range=CLASS_COLORS), legend=None),
    )
    return style_chart(chart)


def reset_filters():
    st.session_state["class_filter"] = [1, 2, 3]
    st.session_state["sex_filter"] = ["female", "male"]
    st.session_state["age_filter"] = AGE_LABELS.copy()


def main():
    st.set_page_config(page_title="Titanic Passenger Insights", page_icon="🚢", layout="wide")
    st.markdown("""
        <style>
        .block-container {max-width:1280px;padding-top:2.4rem;padding-bottom:2.5rem;}
        h1,h2,h3 {color:#18344A;letter-spacing:-.035em;}
        h1 {font-size:2.65rem !important;}
        [data-testid="stSidebar"] {background:#EEF4F4;}
        [data-testid="stMetric"] {background:white;border:1px solid #E2E9ED;
          border-radius:12px;padding:1rem 1.15rem;border-top:3px solid #168B83;}
        [data-testid="stMetricLabel"] {color:#536776;}
        [data-testid="stMetricValue"] {color:#18344A;}
        button[data-baseweb="tab"] {font-weight:600;padding-left:1rem;padding-right:1rem;}
        </style>
    """, unsafe_allow_html=True)

    try:
        raw_df = load_data()
        cleaned_df, transformed_df, details = prepare_data(raw_df)
    except (OSError, ValueError, IndexError) as error:
        st.error(f"The Titanic dataset could not be loaded: {error}")
        st.info("Place titanic.csv in the same folder as app.py and restart the app.")
        st.stop()

    filter_defaults = {
        "class_filter": [1, 2, 3],
        "sex_filter": ["female", "male"],
        "age_filter": AGE_LABELS.copy(),
    }
    for key, default in filter_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        st.title("Passenger filters")
        st.caption("Choose a passenger group to explore.")
        st.button("Reset filters", on_click=reset_filters, width="stretch")
        classes = st.multiselect("Passenger class", [1, 2, 3],
                                 format_func=lambda value: CLASS_LABELS[value], key="class_filter")
        sexes = st.multiselect("Sex", ["female", "male"],
                              format_func=str.capitalize, key="sex_filter")
        ages = st.multiselect("Age group", AGE_LABELS, key="age_filter")
        filtered_df = filter_data(transformed_df, classes, sexes, ages)
        st.divider()
        st.metric("Selected passengers", f"{len(filtered_df):,}")
        st.caption(f"{len(filtered_df) / len(raw_df):.1%} of {len(raw_df):,} records")
        st.caption("Age groups use completed years. Missing ages use the dataset median.")

    st.caption("PROGRAMMING FOR DATA SCIENCE  /  TITANIC DATASET")
    st.title("Titanic Passenger Insights")
    st.markdown("Explore survival patterns across passenger groups.")
    st.caption(f"{len(raw_df):,} passenger records · {raw_df.shape[1]} original columns · Shared filters across all charts")

    overview, explore, preprocessing, explorer = st.tabs(["Overview", "Explore", "Preprocessing", "Data Explorer"])

    with overview:
        st.subheader("Your selection at a glance")
        kpis = st.columns(4)
        kpis[0].metric("Passengers in selection", f"{len(filtered_df):,}")
        kpis[1].metric("Survivors", f"{int(filtered_df['survived'].sum()):,}")
        kpis[2].metric("Survival rate", f"{filtered_df['survived'].mean():.1%}" if len(filtered_df) else "—")
        kpis[3].metric("Average fare", f"£{filtered_df['fare'].mean():.2f}" if len(filtered_df) else "—")
        st.write("")
        if filtered_df.empty:
            st.info("No passengers match these filters. Select at least one option in each filter or reset them.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.subheader("Survival counts")
                st.altair_chart(survival_counts_chart(filtered_df), width="stretch")
            with right:
                st.subheader("Survival rate by passenger class")
                st.altair_chart(rate_chart(filtered_df, "pclass"), width="stretch")
        st.caption("Rates describe the selected records in this dataset only; associations do not establish causation.")

    with explore:
        st.subheader("Compare passenger groups")
        st.caption("Hover over a bar for its group size and survival rate.")
        if filtered_df.empty:
            st.info("Broaden the filters to display comparison charts.")
        else:
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("#### Survival rate by class")
                st.altair_chart(rate_chart(filtered_df, "pclass"), width="stretch")
                st.markdown("#### Age distribution")
                st.altair_chart(age_histogram(filtered_df), width="stretch")
                imputed_count = int(raw_df.loc[filtered_df.index, "age"].isna().sum())
                st.caption(f"Five-year bins; {imputed_count} selected ages were filled with the median ({details['age_median']:g} years).")
            with right:
                st.markdown("#### Survival rate by sex")
                st.altair_chart(rate_chart(filtered_df, "sex"), width="stretch")
                st.markdown("#### Fare distribution by class")
                st.altair_chart(fare_chart(filtered_df), width="stretch")
                st.caption("Box plots show the median, middle 50%, whiskers and outlying fares.")
        st.markdown("#### One observed finding")
        st.info(descriptive_finding(filtered_df))

    with preprocessing:
        st.subheader("From raw records to useful features")
        st.caption("Preparation is performed on the full dataset before filtering. Examples below use all records.")
        with st.expander("1 · Handle missing values", expanded=True):
            st.write(f"Missing ages are filled with the median ({details['age_median']:g}); missing embarkation ports use the most frequent category ({details['embarked_mode']}).")
            st.dataframe(details["missing"], width="stretch")
            changed = raw_df.index[raw_df["age"].isna() | raw_df["embarked"].isna()][:8]
            preview = pd.DataFrame({
                "age_before": raw_df.loc[changed, "age"], "age_after": cleaned_df.loc[changed, "age"],
                "embarked_before": raw_df.loc[changed, "embarked"], "embarked_after": cleaned_df.loc[changed, "embarked"],
            })
            st.dataframe(preview, width="stretch")
            deck_missing = int(raw_df["deck"].isna().sum()) if "deck" in raw_df else 0
            st.caption(f"Deck is excluded from analysis because {deck_missing} values are missing. Raw records remain unchanged.")
        with st.expander("2 · Min-Max normalization"):
            st.write("Age and fare are each scaled to 0–1 using their own full-dataset minimum and maximum; the original columns are retained.")
            st.latex(r"x_{\mathrm{norm}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}}")
            st.dataframe(transformed_df[["age", "age_norm", "fare", "fare_norm"]].head(10), width="stretch")
            st.caption("Normalization is calculated once before filters, so a passenger's normalized value stays consistent across selections.")
        with st.expander("3 · Age discretization"):
            st.write("Age in years is converted into five ordered categories after missing ages are filled. Completed-year boundaries also include fractional infant ages.")
            st.dataframe(pd.DataFrame({"Age group": AGE_LABELS, "Worksheet range": ["0–12", "13–18", "19–35", "36–60", "61+"],
                                      "Exact interval": ["0 ≤ age < 13", "13 ≤ age < 19", "19 ≤ age < 36", "36 ≤ age < 61", "61 ≤ age"]}), hide_index=True)
            st.dataframe(transformed_df[["age", "age_group"]].head(10), width="stretch")
        with st.expander("4 · Dummy coding"):
            st.write("Sex and embarked become 0/1 indicator columns: 1 indicates membership in that category. Readable source categories are retained for exploration.")
            left, right = st.columns([1, 2])
            with left:
                st.markdown("**Before**")
                st.dataframe(cleaned_df[["sex", "embarked"]].head(8), width="stretch")
            with right:
                st.markdown("**After**")
                st.dataframe(transformed_df[details["dummy_columns"]].head(8), width="stretch")
            st.caption("Survived remains the original 0/1 outcome. Dummy columns remain 0/1 and are not normalized.")

    with explorer:
        st.subheader("Inspect the passenger records")
        st.caption("Sidebar filters select the same passenger indices in every view. In Raw, missing values are preserved; age-group membership uses the cleaned age.")
        view = st.radio("Dataset view", ["Raw", "Cleaned", "Transformed"], horizontal=True, key="data_view")
        source = {"Raw": raw_df, "Cleaned": cleaned_df, "Transformed": transformed_df}[view]
        defaults = FIELDS + (["age_group", "age_norm", "fare_norm"] if view == "Transformed" else [])
        columns = st.multiselect("Columns to display", source.columns.tolist(), default=defaults, key=f"columns_{view}")
        st.caption(f"{len(filtered_df):,} rows after filters · {len(columns)} displayed columns · {view.lower()} view")
        selected_view = source.loc[filtered_df.index, columns]
        if columns:
            st.dataframe(selected_view, width="stretch", height=380)
            st.download_button("Download this selection as CSV", selected_view.to_csv(index=True, index_label="passenger_index").encode("utf-8"),
                               file_name=f"titanic_{view.lower()}_selection.csv", mime="text/csv")
        else:
            st.info("Choose at least one column to display the table.")
        with st.expander("Dataset inspection · Task 1"):
            st.write(f"Actual raw shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns.")
            st.markdown("**Column names, data types and missing values**")
            inspection = pd.DataFrame({"Data type": raw_df.dtypes.astype(str), "Missing values": raw_df.isna().sum()})
            inspection.index.name = "Column name"
            st.dataframe(inspection, width="stretch")
            st.markdown("**First five raw records**")
            st.dataframe(raw_df.head(5), width="stretch")
            st.caption("The .loc selector is used to select the six analysis columns and to align filtered records across views.")

    st.divider()
    st.caption("Source: Seaborn's Titanic dataset · Observational exploration · No prediction model")


if __name__ == "__main__":
    main()

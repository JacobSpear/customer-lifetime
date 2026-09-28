import pytest
from csurv.features import build_features

@pytest.fixture(scope="module")
def df():
    return build_features("data/demo.db")

def test_no_nan_in_targets(df):
    assert not df["time"].isna().any()
    assert not df["event"].isna().any()

def test_positive_duration(df):
    # Allow time=0 only for censored records on the data cutoff date
    bad = df[(df["time"] == 0) & ~((df["event"] == 0) & (df["visit_date"] == df["visit_date"].max()))]
    if len(bad)>0:
        print(bad[['customer_id','visit_id']])
    assert len(bad) == 0, f"{len(bad)} rows with duration=0 not on cutoff date"

def test_event_is_binary(df):
    assert set(df["event"].unique()) == {0, 1}

def test_one_censored_per_customer(df):
    censored_counts = df[df["event"] == 0].groupby("customer_id").size()
    assert (censored_counts == 1).all()

def test_last_visit_is_censored(df):
    last_visits = df.sort_values("gap_num").groupby("customer_id").last()
    assert (last_visits["event"] == 0).all()
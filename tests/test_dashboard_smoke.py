from src.pages.home import load_df

def test_load_df_not_empty():
    df = load_df()
    assert len(df) > 0

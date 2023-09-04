def create_df(columns: tuple, data: list[tuple], spark_session, partitions=1):
    """
    columns = 'top_floor', 'count'
    data = [("5", 3), ("3", 2), ("2", 1), ("10", 1)]
    """
    return spark_session.createDataFrame(data).toDF(*columns).coalesce(partitions)


def compare(expected_df, actual_df):
    if (actual_df is None) | (expected_df is None):
        print()
        print("Check Dataframes")
        return

    print()
    print("expected:")
    expected_df = expected_df.cache()
    expected_df.show()

    print("actual:")
    actual_df = actual_df.cache()
    actual_df.show() if actual_df is not None else print("Actual Dataframe Is None")

    df_compare = expected_df.subtract(actual_df)
    if df_compare.count() > 0:
        print("difference:")
        expected_df.subtract(actual_df).show() if actual_df is not None else print("Actual Dataframe Is None")
def create_df(columns: tuple, data: list[tuple], spark_session, partitions=1):
    """
    columns = 'top_floor', 'count'
    data = [("5", 3), ("3", 2), ("2", 1), ("10", 1)]
    """
    return spark_session.createDataFrame(data).toDF(*columns).coalesce(partitions)


def compare(df1_exp, df2_act):
    print()
    print("expected:")
    df1_exp = df1_exp.cache()
    df1_exp.show()
    print("actual:")
    df2_act = df2_act.cache()
    df2_act.show() if df2_act is not None else print("Actual Dataframe Is None")

    df_compare = df1_exp.subtract(df2_act)
    if df_compare.count() > 0:
        print("difference:")
        df1_exp.subtract(df2_act).show() if df2_act is not None else print("Actual Dataframe Is None")
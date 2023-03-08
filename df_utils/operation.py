from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import StringType, DataType
from functools import partial, reduce


def melt(df, keep_cols, melt_cols, var_name, value_name):
    _vars_and_vals = F.array(*(F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name)) for c in melt_cols))
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))
    cols = keep_cols + [F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]

    return _tmp.select(*cols)


def stack(df: DataFrame, columns: list, var_col: str, var_val: str) -> DataFrame:
    stack_cols: str = ''.join(f"'{c}',{c}," for c in columns)[:-1]  # without last comma
    exp = "stack(" + str(len(columns)) + "," + stack_cols + f") as ({var_col}, {var_val})"
    return df.selectExpr("*", exp).drop(*columns)


def union_all(*dfs, allow_missing_columns: bool = False):
    return reduce(
        partial(DataFrame.unionByName, allowMissingColumns=allow_missing_columns), dfs
    )


def rename_df(df: DataFrame, rename_dict: dict):
    _new_cols = [rename_dict.get(i, i) for i in df.columns]
    return df.toDF(*_new_cols)


def add_missing_columns(df: DataFrame, ref_df: DataFrame) -> DataFrame:
    """Add missing columns from ref_df to df

    Args:
        df (DataFrame): dataframe with missing columns
        ref_df (DataFrame): referential dataframe

    Returns:
        DataFrame: df with additionnal columns from ref_df
    """
    for col in ref_df.schema:
        if col.name not in df.columns:
            df = df.withColumn(col.name, F.lit(None).cast(col.dataType))

    return df


def cast_columns(df: DataFrame, colz: list, new_type: DataType = StringType()) -> DataFrame:
    return df.select([F.col(c).cast(new_type) if c in colz else c for c in df.columns])


def null_safe_sum(col1:str, col2: str) -> F.Column:
    _both_nulls = F.isnull(col1) & F.isnull(col2)
    _sum = F.coalesce(col1, F.lit(0)) + F.coalesce(col2, F.lit(0))
    return F.when(_both_nulls, F.lit(0)).otherwise(_sum)

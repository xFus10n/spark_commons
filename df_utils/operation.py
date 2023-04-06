from pyspark.sql import DataFrame, functions as F, Column
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
        DataFrame: df with additional columns from ref_df
    """
    for col in ref_df.schema:
        if col.name not in df.columns:
            df = df.withColumn(col.name, F.lit(None).cast(col.dataType))

    return df


def cast_columns(df: DataFrame, colz: list, new_type: DataType = StringType()) -> DataFrame:
    return df.select([F.col(c).cast(new_type) if c in colz else c for c in df.columns])


def null_safe_sum(*col_n: str, replace_null=0) -> Column:
    if len(col_n) == 0:
        return F.lit(0)
    else:
        _sum = F.coalesce(col_n[0], F.lit(replace_null))
        for column in col_n[1:]:
            _sum = _sum.__add__(F.coalesce(column, F.lit(replace_null)))
    return _sum


def null_safe_sub(*col_n: str, replace_null=0) -> Column:
    if col_n:
        _sub = F.coalesce(col_n[0], F.lit(replace_null))
        for column in col_n[1:]:
            _sub = _sub.__sub__(F.coalesce(column, F.lit(replace_null)))
        return _sub
    else:
        return F.lit(0)
    # if len(col_n) == 0:
    #     return F.lit(0)
    # else:
    #     _sub = F.coalesce(col_n[0], F.lit(replace_null))
    #     for column in col_n[1:]:
    #         _sub = _sub.__sub__(F.coalesce(column, F.lit(replace_null)))
    # return _sub


def chain_conditions(conditions_list: list[Column], default_value: Column) -> Column:

    if conditions_list:
        return conditions_list[0].otherwise(chain_conditions(conditions_list[1:], default_value))
    else:
        return default_value

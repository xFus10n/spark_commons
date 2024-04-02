from pyspark.sql import DataFrame, functions as F, Column
from pyspark.sql.types import StringType, DataType
from functools import partial, reduce


def melt(df, keep_cols, melt_cols, var_name, value_name):
    """
    Transforms the DataFrame from wide format to long format by melting the specified columns.
    This operation is similar to the 'melt' function in pandas or the 'gather' function in tidyr in R.

    Parameters:
    - df (DataFrame): The input DataFrame to be melted.
    - keep_cols (list): A list of column names that will remain unchanged.
    - melt_cols (list): A list of column names that will be melted into two columns,
      one for variable names and one for values.
    - var_name (str): The name of the new column that will contain the variable names.
      This corresponds to the column headers from the `melt_cols` in the original DataFrame.
    - value_name (str): The name of the new column that will contain the values from the melted columns.

    Returns:
    DataFrame: A melted DataFrame with the columns specified in `keep_cols`,
    along with two new columns: one for the variable names (`var_name`) and one for the values (`value_name`).

    Example:
    Suppose you have the following DataFrame:
    +---+------+------+------+
    | id|  var1|  var2|  var3|
    +---+------+------+------+
    |  1|     A|     B|     C|
    |  2|     D|     E|     F|
    +---+------+------+------+

    And you want to melt `var1`, `var2`, and `var3` columns, keeping `id` column intact. You can call the function like this:

    melted_df = melt(
        df=df,
        keep_cols=['id'],
        melt_cols=['var1', 'var2', 'var3'],
        var_name='variable',
        value_name='value'
    )

    The resulting DataFrame will be:
    +---+--------+-----+
    | id|variable|value|
    +---+--------+-----+
    |  1|    var1|    A|
    |  1|    var2|    B|
    |  1|    var3|    C|
    |  2|    var1|    D|
    |  2|    var2|    E|
    |  2|    var3|    F|
    +---+--------+-----+
    """
    _vars_and_vals = F.array(*(F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name)) for c in melt_cols))
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))
    cols = keep_cols + [F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]

    return _tmp.select(*cols)


def stack(df: DataFrame, columns: list, var_col: str, var_val: str) -> DataFrame:
    """
    Transforms the DataFrame from wide format to long format by stacking the specified columns into two columns:
    one for the variable names (keys) and one for the values. This operation is akin to "melting" or "unpivoting"
    in other data manipulation frameworks.

    Parameters:
    - df (DataFrame): The input Spark DataFrame to be transformed.
    - columns (list): A list of column names in `df` that will be stacked. These columns will be transformed into
      two new columns: one holding the variable names and the other the corresponding values.
    - var_col (str): The name of the new column that will contain the names of the variables from `columns`.
      This acts as the key column in the resulting long-format DataFrame.
    - var_val (str): The name of the new column that will contain the values corresponding to each variable
      name in `var_col`. This acts as the value column in the resulting long-format DataFrame.

    Returns:
    DataFrame: A DataFrame in long format with the original columns specified in `columns` removed and replaced
    by two new columns: `var_col` (variable names) and `var_val` (values). The rest of the DataFrame columns
    remain unchanged.

    Example:
    Given a DataFrame `df`:
    +----+------+------+------+
    | id | var1 | var2 | var3 |
    +----+------+------+------+
    |  1 |    A |    B |    C |
    |  2 |    D |    E |    F |
    +----+------+------+------+

    Calling `stack(df, ['var1', 'var2', 'var3'], 'variable', 'value')` will return:
    +---+--------+-----+
    | id|variable|value|
    +---+--------+-----+
    |  1|    var1|    A|
    |  1|    var2|    B|
    |  1|    var3|    C|
    |  2|    var1|    D|
    |  2|    var2|    E|
    |  2|    var3|    F|
    +---+--------+-----+

    Note:
    - Uses native SQL function
    """
    stack_cols: str = ''.join(f"'{c}',{c}," for c in columns)[:-1]
    exp = "stack(" + str(len(columns)) + "," + stack_cols + f") as ({var_col}, {var_val})"
    return df.selectExpr("*", exp).drop(*columns)


def union_all(*dfs, allow_missing_columns: bool = False):
    """
    Performs a union of multiple DataFrames by column names, with an option to allow missing columns.

    Parameters:
    - *dfs: A variable number of DataFrame objects to be unioned.
    - allow_missing_columns (bool, optional): If True, allows union operation on DataFrames with non-identical schemas,
      filling missing columns with nulls. Defaults to False.

    Returns:
    DataFrame: A DataFrame resulting from the union of all input DataFrames.

    Raises:
    ValueError: If no DataFrames are provided or if any of the provided arguments is not a DataFrame.

    Example:
    Assuming df1, df2, and df3 are Spark DataFrames with potentially different schemas:

    df1.show()
    +---+----+
    | id|name|
    +---+----+
    |  1|   A|
    +---+----+

    df2.show()
    +---+----+------+
    | id|name|salary|
    +---+----+------+
    |  2|   B| 10000|
    +---+----+------+

    unioned_df = union_all(df1, df2, allow_missing_columns=True)
    unioned_df.show()
    +---+----+------+
    | id|name|salary|
    +---+----+------+
    |  1|   A|  null|
    |  2|   B| 10000|
    +---+----+------+

    Note:
    - The order of columns in the resulting DataFrame is determined by the order of columns in the first DataFrame.
    """
    if not dfs:
        raise ValueError("At least one DataFrame must be provided for union operation.")
    if not all(isinstance(df, DataFrame) for df in dfs):
        raise ValueError("All provided arguments must be DataFrames.")
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
    """
    Casts specified columns of a PySpark DataFrame to a new data type.

    Parameters:
    - df (DataFrame): The input PySpark DataFrame whose columns are to be cast.
    - colz (list): A list of column names (strings) in the DataFrame that should be cast to the new data type.
    - new_type (DataType, optional): The target data type to which the specified columns should be cast.
      This should be a PySpark SQL DataType (e.g., StringType(), IntegerType()). The default is StringType().

    Returns:
    - DataFrame: A new DataFrame with the specified columns cast to the new data type. Other columns
      remain unchanged.

    Example:
    --------
    df = spark.createDataFrame([("1", "200"), ("2", "300"), ("3", "400")], ["id", "value"])
    casted_df = cast_columns(df, ["value"], IntegerType())
    casted_df.printSchema()
    root
     |-- id: string (nullable = true)
     |-- value: integer (nullable = true)
    """
    return df.select([F.col(c).cast(new_type) if c in colz else c for c in df.columns])


def null_safe_sum(*col_n: str, replace_null=0) -> Column:
    """
    Performs a null-safe summation operation on a sequence of column names or Column expressions.

    Parameters:
    - *col_n (str): Variable length argument list specifying the column names or Column expressions
      to be summed. The summation is performed from left to right.
    - replace_null (int or float, optional): The value to use in place of null values during
      summation. Defaults to 0.

    Returns:
    - Column: A PySpark Column object resulting from the null-safe summation of the provided columns.
      If no columns are provided, it returns a literal column with the `replace_null` value.

    Example:
    --------
    df = spark.createDataFrame([(100, None, 50), (None, 50, 50), (30, 20, None)], ["col1", "col2", "col3"])
    df = df.withColumn("summed", null_safe_sum(col("col1"), col("col2"), col("col3"), replace_null=0))
    df.show()
    +----+----+----+------+
    |col1|col2|col3|summed|
    +----+----+----+------+
    | 100|null|  50| 150.0|
    |null|  50|  50| 100.0|
    |  30|  20|null|  50.0|
    +----+----+----+------+
    """
    if col_n:
        _sum = F.coalesce(col_n[0], F.lit(replace_null))
        for column in col_n[1:]:
            _sum = _sum.__add__(F.coalesce(column, F.lit(replace_null)))
        return _sum
    else:
        return F.lit(replace_null)


def null_safe_sub(*col_n: str, replace_null=0) -> Column:
    """
    Performs a null-safe subtraction operation on a sequence of column names or Column expressions.

    Parameters:
    - *col_n (str): Variable length argument list specifying the column names or Column expressions
      to be subtracted. The subtraction is performed from left to right.
    - replace_null (int or float, optional): The value to use in place of null values during
      subtraction. Defaults to 0.

    Returns:
    - Column: A PySpark Column object resulting from the null-safe subtraction of the provided columns.
      If no columns are provided, it returns a literal column with the `replace_null` value.

    Example:
    --------
    df = spark.createDataFrame([(100, None), (None, 50), (30, 20)], ["col1", "col2"])
    df = df.withColumn("subtracted", null_safe_sub(col("col1"), col("col2"), replace_null=0))
    df.show()
    +----+----+-----------+
    |col1|col2|subtracted |
    +----+----+-----------+
    | 100|null|      100.0|
    |null|  50|      -50.0|
    |  30|  20|       10.0|
    +----+----+-----------+
    """
    if col_n:
        _sub = F.coalesce(col_n[0], F.lit(replace_null))
        for column in col_n[1:]:
            _sub = _sub.__sub__(F.coalesce(column, F.lit(replace_null)))
        return _sub
    else:
        return F.lit(replace_null)


def chain_conditions(conditions_list: list[Column], default_value: Column) -> Column:
    """
    Chains together a sequence of PySpark Column conditions with an optional default value.

    This function is designed to concatenate a list of PySpark Column conditions using the
    otherwise chaining method, creating a cascading set of conditions similar to an 'if-elif-else'
    statement in traditional programming. If none of the conditions in the list are satisfied,
    the default value is returned.

    Parameters:
    - conditions_list (list[Column]): A list of PySpark Column conditions to be chained together.
    - default_value (Column): The default value to be returned if none of the conditions are met.

    Returns:
    - Column: A PySpark Column object representing the chained conditions and default value.

    Example:
    --------
    df = spark.createDataFrame([(1, 'a'), (2, 'b'), (3, 'c')], ["id", "value"])
    condition1 = when(col("id") == 1, "One")
    condition2 = when(col("id") == 2, "Two")
    condition3 = when(col("id") == 3, "Three")
    default_condition = "Other"
    result = df.withColumn("result", chain_conditions([condition1, condition2, condition3], default_condition))
    result.show()
    +---+-----+------+
    | id|value|result|
    +---+-----+------+
    |  1|    a|   One|
    |  2|    b|   Two|
    |  3|    c| Three|
    +---+-----+------+

    Notes:
    ------
    - Ensure that the conditions provided in the `conditions_list` are mutually exclusive to avoid
      ambiguous results. If multiple conditions can be satisfied simultaneously, only the first
      satisfied condition will be considered.
    - The default value is returned when none of the conditions in the `conditions_list` are met.
    """
    if conditions_list:
        return conditions_list[0].otherwise(chain_conditions(conditions_list[1:], default_value))
    else:
        return default_value

def with_columns(cols, df: DataFrame, f: callable) -> DataFrame:
    """
    :param cols: a list of column names to transform
    :param df: an input DataFrame
    :param f: a function to be applied on each column name in cols. It should return a Column
    :return: DataFrame with the transformations applied
    """
    # Select all columns from the existing DataFrame
    # and apply the transformation function to the specified columns
    selected_cols = [F.col(c) for c in df.columns if c not in cols] + [f(c).alias(c) for c in cols]
    return df.select(*selected_cols)
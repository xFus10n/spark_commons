import logging
import pytest
from pyspark_test import assert_pyspark_df_equal
from pyspark.sql import SparkSession, functions as F
from tests.utilz.df_test_helper import compare
from tests.operation_tests.operation_test_data import (
    get_test_data,
    get_expect_data,
    get_rename_dict,
    get_test_rename_data
)
from df_utils.operation import (
    melt,
    stack,
    union_all,
    rename_df,
    null_safe_sum,
    null_safe_sub,
    add_missing_columns
)


def quiet_py4j():
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_session(request):
    spark_session = SparkSession.builder.getOrCreate()
    request.addfinalizer(lambda: spark_session.stop())
    quiet_py4j()
    return spark_session


def test_add_missing_columns(spark_session):
    # arrange
    test_df = get_test_data(spark_session).cache()

    # act
    expected_df = test_df.withColumn("test", F.lit(None))
    actual_df = add_missing_columns(test_df, expected_df)

    # assert
    compare(expected_df, actual_df)
    assert_pyspark_df_equal(expected_df, actual_df)


def test_null_safe_sum(spark_session):
    # arrange
    test_df = get_test_data(spark_session).cache()

    # act
    test_df1 = test_df.withColumn("Y1", F.lit(None))
    test_df2 = test_df1.withColumn("Y2", F.lit(None))
    output_df = test_df2.withColumn("sum", F.round(null_safe_sum("Y1", "Y2", "Y3"), 2)).select("sum")
    expected_df = test_df.select("Y3").withColumnRenamed("Y3", "sum")

    # assert
    compare(expected_df, output_df)
    assert_pyspark_df_equal(expected_df, output_df)


def test_null_safe_sub(spark_session):
    # arrange
    test_df = get_test_data(spark_session).cache()

    # act
    test_df1 = test_df.withColumn("Y1", F.lit(None))
    output_df = test_df1.withColumn("sub", F.round(null_safe_sub("Y3", "Y2", "Y1"), 2)).select("sub")
    expected_df = test_df.withColumn("sub", F.round(F.col("Y3") - F.col("Y2"), 2)).select("sub")

    # assert
    compare(expected_df, output_df)
    assert_pyspark_df_equal(expected_df, output_df)


def test_rename_dict(spark_session):
    # arrange
    test_df = get_test_data(spark_session)
    expected_df = get_test_rename_data(spark_session)

    # act
    output_df = rename_df(test_df, get_rename_dict())

    # assert
    compare(expected_df, output_df)
    assert_pyspark_df_equal(expected_df, output_df)


def test_melt_function(spark_session):
    # arrange
    test_df = get_test_data(spark_session)
    expected_df = get_expect_data(spark_session)

    # act
    output_df = melt(test_df, ["col1", "col2", "col3"], ["Y1", "Y2", "Y3"], "category", "value").sort(F.col("value"))

    # assert
    compare(expected_df, output_df)
    assert_pyspark_df_equal(expected_df, output_df)


def test_stack_function(spark_session):
    # arrange
    test_df = get_test_data(spark_session)
    expected_df = get_expect_data(spark_session)

    # act
    output_df = stack(test_df, ["Y1", "Y2", "Y3"], "category", "value").sort(F.col("value"))

    # assert
    compare(expected_df, output_df)
    assert_pyspark_df_equal(expected_df, output_df)


def test_union(spark_session):
    # arrange
    test_df = get_test_data(spark_session)
    expected_df = get_test_data(spark_session)

    # act
    actual_df1 = test_df.where(F.col("col3") == "89EB")
    actual_df2 = test_df.where(F.col("col3") == "89EK")
    actual_df3 = test_df.where(F.col("col3") == "89EN")
    output_df = union_all(actual_df1, actual_df2, actual_df3)

    # assert
    compare(expected_df, output_df)
    assert_pyspark_df_equal(expected_df, output_df)
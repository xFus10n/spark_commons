import logging
import pytest
from pyspark_test import assert_pyspark_df_equal
from pyspark.sql import SparkSession, functions as F
from tests.utilz.df_test_helper import compare, create_df
from df_utils.operation import melt, stack


def quiet_py4j():
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_session(request):
    spark_session = SparkSession.builder.getOrCreate()
    request.addfinalizer(lambda: spark_session.stop())
    quiet_py4j()
    return spark_session


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


def get_test_data(spark_session):
    columns = ("col1", "col2", "col3", "Y1", "Y2", "Y3")
    data = [(4000521, "BOF_I2189490CA-20", "89EB", 1000.01, 2000.02, 3000.03),
            (4000521, "BOF_I2189490CA-20", "89EK", 4000.04, 5000.05, 6000.06),
            (4000521, "BOF_I2189490CA-20", "89EN", 7000.07, 8000.08, 9000.09)]
    return create_df(columns, data, spark_session)


def get_expect_data(spark_session):
    columns = ("col1", "col2", "col3", "category", "value")
    data = [(4000521, "BOF_I2189490CA-20", "89EB", "Y1", 1000.01),
            (4000521, "BOF_I2189490CA-20", "89EB", "Y2", 2000.02),
            (4000521, "BOF_I2189490CA-20", "89EB", "Y3", 3000.03),
            (4000521, "BOF_I2189490CA-20", "89EK", "Y1", 4000.04),
            (4000521, "BOF_I2189490CA-20", "89EK", "Y2", 5000.05),
            (4000521, "BOF_I2189490CA-20", "89EK", "Y3", 6000.06),
            (4000521, "BOF_I2189490CA-20", "89EN", "Y1", 7000.07),
            (4000521, "BOF_I2189490CA-20", "89EN", "Y2", 8000.08),
            (4000521, "BOF_I2189490CA-20", "89EN", "Y3", 9000.09)]
    return create_df(columns, data, spark_session)

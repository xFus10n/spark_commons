import logging
import pytest
from pyspark_test import assert_pyspark_df_equal
from pyspark.sql import SparkSession


def quiet_py4j():
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def spark_session(request):
    spark_session = SparkSession.builder.getOrCreate()
    request.addfinalizer(lambda: spark_session.stop())
    quiet_py4j()
    return spark_session


def test_top_floor_extract(spark_session):
    # arrange
    test_df = get_output_for_top_floor(spark_session)
    expected_df = get_output_for_top_floor(spark_session)

    # act
    # output_df = main.set_top_floor(test_df)
    # actual_df = output_df.select(main.floor, main.top_floor) if hasattr(output_df, main.top_floor) else None

    # assert
    compare(expected_df, test_df)
    assert_pyspark_df_equal(expected_df, test_df)


def get_output_for_top_floor(spark_session):
    columns = 'top_floor', 'count'
    data = [("5", 3), ("3", 2), ("2", 1), ("10", 1)]
    return spark_session.createDataFrame(data).toDF(*columns)


def compare(df1_exp, df2_act):
    print()
    print("expected:")
    df1_exp.show()
    print("actual:")
    df2_act.show() if df2_act is not None else print("Actual Dataframe Is None")
    print("difference:")
    df1_exp.subtract(df2_act).show() if df2_act is not None else print("Actual Dataframe Is None")
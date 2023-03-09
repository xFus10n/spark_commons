from tests.utilz.df_test_helper import create_df


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


def get_rename_dict():
    return {"col1": "measure1", "col2": "measure2", "col3": "measure3"}


def get_test_rename_data(spark_session):
    columns = ("measure1", "measure2", "measure3", "Y1", "Y2", "Y3")
    data = [(4000521, "BOF_I2189490CA-20", "89EB", 1000.01, 2000.02, 3000.03),
            (4000521, "BOF_I2189490CA-20", "89EK", 4000.04, 5000.05, 6000.06),
            (4000521, "BOF_I2189490CA-20", "89EN", 7000.07, 8000.08, 9000.09)]
    return create_df(columns, data, spark_session)
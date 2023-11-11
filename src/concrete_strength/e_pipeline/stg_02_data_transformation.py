import os
import sys

from concrete_strength import logging, CustomException
from concrete_strength.c_config.configuration import ConfigurationManager
from concrete_strength.d_components.data_transformation import DataTransformation


STAGE_NAME = "DATA -- TRANSFORMATION -- STAGE"


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config)
        data_transformation.train_test_spliting()


if __name__ == '__main__':
    try:
        logging.info(
            f"\n\nx==========x\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = DataTransformationPipeline()
        obj.main()
        logging.info(
            f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x\n\n")
    except Exception as e:
        logging.exception(CustomException(e, sys))
        raise CustomException(e, sys)

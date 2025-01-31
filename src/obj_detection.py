########################################################
#       Object Detection
# Reference : https://muhammadrizwanmunawar.medium.com/ultralytics-yolo11-object-detection-and-instance-segmentation-88ef0239a811
########################################################
import logging
import os
import argparse
from ultralytics import YOLO


def main(input_file):

    # Log file
    path = "log"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory 'log' is created!")
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    filename='log/obj_detection' + date_time + '.log'
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Print input file
    print(f'Input file: {input_file}')

    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Run inference on the source
    results = model(input_file)  # list of Results objects

    # Loop through the results and display each one
    for result in results:
        result.show()  # Show the image with detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str,
                        required=True, help="input dataset")
    args = parser.parse_args()
    print(args.input_file)
    main(args.input_file)
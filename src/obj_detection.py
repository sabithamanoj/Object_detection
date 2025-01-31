########################################################
#       Object Detection
########################################################
import logging
import os
import argparse
from ultralytics import YOLO
import cv2


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

    # To save the findings
    input_file_name = os.path.basename(input_file)
    if results:
        img_with_boxes = results[0].plot()  # Draw all bounding boxes on the image
        save_path = 'outputs/output_' + input_file_name # Define save path
        print(f"save_path: {save_path}")
        cv2.imwrite(save_path, img_with_boxes)  # Save image
        print(f"Image saved at: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str,
                        required=True, help="input dataset")
    args = parser.parse_args()
    print(args.input_file)
    main(args.input_file)
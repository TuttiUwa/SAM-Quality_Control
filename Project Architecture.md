# Project Architecture

- **data**: A directory contaning our data files (images).
    - **Data_Transformation**: Contains the train, test, and validation data, as well as the data.yaml file.
    - **PCB_xray_dataset**: Contains the raw data.

- **model**: this directory contains our YOLO detection and SAM segmentation models. They're ready to be used.

- **templates**: Contains the basic HTML code we need to run our app.
    - **base.html**: The base architecture with free slots waiting to be filled with some content
    - **index.html**: Our app entry point, extending `base.html`.

- **app.py**: where are defined the route functions of our app.

- **detection.py**: contains the YOLO detection script.

- **segmentation.py**: contains the SAM segmentation script.

- **report.py**: contains the report generation script.


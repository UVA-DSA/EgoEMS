# Depth Estimation Code

This folder contains code for depth estimation using various techniques. The primary focus is on evaluating the quality of CPR (Cardiopulmonary Resuscitation) using vision-based methods.

## Contents

- `estimate.py`: Main script for performing depth estimation.
- `extract_depth.py`: `.mkv` file reading helpers.
- `format_final_data.py`: Script for formatting the final data.

## Dependencies

This project relies on several dependencies, including Open3D. Please note that there are some known shortfalls with Open3D dependencies that may affect the performance and compatibility of the code.

## Known Issues

1. **Open3D Dependencies**: The code depends on Open3D, which may have compatibility issues with certain versions of Python and other libraries. Ensure you have the correct versions installed to avoid any runtime errors.

2. **Hard-Coded Values**: The `format_final_data.py` script contains hard-coded values for paths and formats. This may require manual adjustments based on your specific setup and data organization.

## Installation

The code depends on `opencv-python`, `PyTorch`, and `Open3D`. Follow the instructions on the PyTorch website and Open3D website to install each. Officially the Kinect Sensor SDK can only be installed on Ubuntu 18.04, but unofficially it should work on 20.04 but you have to configure the 18.04 Microsoft Ubuntu repository.

## Usage

To run the CPR estimation pipeline, it takes a few workarounds. 

```bash
python estimate.py [(train, val, test) index] [start_index] [end_index (optional)]
```

If you run into issues with ERROR CODE 204 when reading `.mkv` files, it's likely because your PC/Server is reaching an I/O bottleneck. Configuring `run_estimation.sh` may help with that.

## Contact

For any questions or issues with this code feel free to email cqa3ym@virginia.edu

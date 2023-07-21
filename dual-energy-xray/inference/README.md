# Dual Energy Heart Segmentation

1. Install environment
   ```sh
   conda create -n cvs python==3.8
   conda activate cvs
   ```
2. Install required packages
   ```sh
   pip install -r requirements.txt
   ```
3. Prepare inputs folder
   ```
   inputs
   |- image_front_combined.dcm
   |- image_front_soft.dcm
   |- image_front_hard.dcm
   ```
4. Run inference, where <inputs> is directory in the step above, and <output> is the file path where output is saved
   ```sh
   python inference.py --inputs <inputs> --output <output>
   ```
   For example
   ```sh
   python inference.py --inputs samples/006_20221109 --output inference.png
   ```

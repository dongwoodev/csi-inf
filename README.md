# CSI DATA INFERENCE
## Introduce 
1. `collect` : Collecting data Module for training.
2. `labeling` : CSI labeling based on images.
3. `trains` : Retruns Pre-processing and Bi-LSTM based learning Model for Sit/Stand Training
4. `inference` : Return Labelling results inferred by Sit/Stand Classify Model using Collected CSI data & Visualize Labeling through images


## Collect Module
<figure>
  <img src="https://github.com/dongwoodev/csi-inf/blob/main/assets/compact_acquire.png", alt="csi_collect_image_compact.py" width=600px>
  <figcaption>CSI-Collecting-image-Compact</figcaption>
</figure>



- `FIRST_README.md`
  - A manual on the environment settings you need to build to collect data
- `csi_collect_recog.py`
  - Press the START button manually, or at a certain time, the START button is activated to collect data.
- `csi_collect_recog_auto.py`
  - When a person is automatically recognized or at a certain time, a module that collects CSI data and image data with two cameras, simultaneously deducing the number and location of people.
- `csi_collect_iamge_compact.py`
  - A module that collects CSI data and image data with two cameras when a person is recognized or at a specific time (Not inferenced image)
- `csi_collect_inf.py` : Infer Sit, stand without image collection (model required)

||collect csi|collect image|human detection info.(skeleton, bbox: img, csv) |passive collect|auto collection(human detection based)|
|---|:---:|:---:|:---:|:---:|:---:|
|csi_collect_recog|O|O|X|O|X|
|csi_collect_recog_auto|O|O|O|O|O|
|**csi_collect_compact**|O|O|X|O|O|
|csi_collect_inf|O|X|X|X|X|

### Reference
- https://github.com/espressif/esp-csi
- https://github.com/espressif/esp-idf
- https://github.com/ultralytics/ultralytics

These codes are created **for research purposes**. If you use them as an application, you can get an error. When you exit the program, you can force `Ctrl+C`.

## Labeling




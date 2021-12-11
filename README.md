# Forehead Extraction and Matching
The use of forehead (creases) as biometric is presented in the codes. The project consists of 2 parts:
1. Forehead Extraction - I trained a YoloV4 model to detect forehead in face image(better use single face at a time)
2. Forehead matching - It matches two forehead images and gives out similarity as the euclideon distance i.e. minimum for same image and large for dissimilar forehead images.

The extraction network is simple YoloV4 while the matching network is a Siamese network trained using Contrastive loss.

## Using the codes
### 1. forehead_detector.py -
It takes input a face image and results a bounding box over that face (forehead). To use it <br />
i. Keep the yolov4.weights and yolov4.cfg files into the same directory. <br />
ii. You need to provide the path to the image.<br />
#### example:<br />
python forehead_detector.py (image_path)<br />
python forehead_detector.py 1.jpg<br />

### 2. forehead_matcher.py -
It takes input the Gpu setting, path to 1st image (forehead) and path to 2nd image(forehead).It gives euclidean distance between the images. To use it<br />
i. Keep the siamese_model.h5 file in same directory.<br />
ii. You need to provide GPU setting i.e 1 if GPU available else 0, path to 1st image and path to 2nd image.<br />
#### example :<br />
python forehead_matcher.py 1(gpu) (pth_1st_image) (pth_2nd_image)<br />
python forehead_matcher.py 1 1_fore.jpg 2_fore.jpg<br />
    
### 3. cropping_and_matching.py -
It combines both the above codes. It takes path of the 2 images(face images) and crops and gives out the euclidean distance. To use it<br />
i. Keep yolov4.weights, yolov4.cfg and siamese_model.h5 files in same directory.<br />
ii. You need to provide Gpu setting, path to 1st face image and path to 2nd face image.<br />
#### example :<br />
python cropping_and_matching.py 1(gpu) (path_1st_img) (path_2nd_img)<br />
python cropping_and_matching.py 1 1_face.jpg 2_face.jpg<br />
      
## Downloading the weights and configuration files
[files available at](https://drive.google.com/drive/folders/1v-KUc_44wzBjckMHSld6zaAZzAuRewwO?usp=sharing)
     

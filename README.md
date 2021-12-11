# Forehead Extraction and Matching
The use of forehead (creases) as biometric is presented in the codes. The project consists of 2 parts:
1. Forehead Extraction - I trained a YoloV4 model to detect forehead in face image(better use single face at a time)
2. Forehead matching - It matches two forehead images and gives out similarity as the euclideon distance i.e. minimum for same image and large for dissimilar forehead images.

The extraction network is simple YoloV4 while the matching network is a Siamese network trained using Contrastive loss.

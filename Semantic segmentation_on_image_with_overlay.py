import pixellib
import cv2
from pixellib.semantic import semantic_segmentation

seg_img = semantic_segmentation()
seg_img.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
seg_img.segmentAsPascalvoc("Image.jpg", output_image_name = "output_image.jpg", overlay = True)

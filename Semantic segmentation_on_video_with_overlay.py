import pixellib
from pixellib.semantic import semantic_segmentation

seg_vid = semantic_segmentation()
seg_vid.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
seg_vid.process_video_pascalvoc("City.mp4",  overlay = True, frames_per_second= 15, output_video_name="output_video.mp4") # Change the input filename accordingly

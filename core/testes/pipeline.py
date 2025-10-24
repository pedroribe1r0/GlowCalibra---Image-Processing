import cv2
import numpy as np
from pathlib import Path
from ..src.IO.file_management import load_image, save_image, export_csv
from ..src.processing.filters import blur, laplacian
from ..src.processing.logical_ops import bitwise_not
from ..src.processing.segmentation import apply_circular_roi, color_treatment, segment_components, thresholdBinary, find_center, grayScale, applyMask, convertToHSV, createHSVMask, thresholdOtsu
from ..src.processing.morphology import closing
from ..src.post_processing.count_drops import count_drops
from ..src.post_processing.find_proportion import find_proportion

DIAMETER_CM = 5

def pipeline():
    image = load_image("IMG_20250507_145742359_HDR.jpg")

    mask = color_treatment(image)
    mask = grayScale(mask)
    save_image(mask, "color_treatment.jpg")
    mask = blur(mask, (11, 11))
    mask = thresholdBinary(mask)

    save_image(mask, "threshold_binary.jpg")
    mask = blur(mask, (9, 9))
    mask = bitwise_not(mask)
    save_image(mask, "bitwise_not.jpg")

    mask = segment_components(mask)
    

    margin = 20
    center, radius = find_center(mask, margin)
    proportion = find_proportion(radius_cm=(DIAMETER_CM/2), radius_px=radius)
    image = apply_circular_roi(image, radius, center, margin)
    save_image(image, "final_mask.jpg")

    image = convertToHSV(image)
    hsvMask = createHSVMask(image, [90, 50, 50], [160, 255, 255])
    image = applyMask(image, hsvMask)
    
    image = grayScale(image)
    
    image = thresholdOtsu(image)
    save_image(image, "final_threshold.jpg")
    

    image = closing(image)
    save_image(image, "final_image.jpg")

    mask, vis, data = count_drops(image, proportion)
    #save_image(mask, "data_mask.jpg")
    save_image(vis, "vis.jpg")
    export_csv("resultados.csv", data)



if __name__ == "__main__":
    pipeline()

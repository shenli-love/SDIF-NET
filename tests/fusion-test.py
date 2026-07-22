import cv2
import numpy as np



def gradient(img):

    gx=cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        3
    )

    gy=cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        3
    )


    return np.sqrt(
        gx**2+gy**2
    )



def robust_ir_vis_fusion(
        ir,
        vis,
        detail_threshold=10
):


    # =====================
    # 1.
    # 可见光灰度
    # =====================

    vis_gray=cv2.cvtColor(
        vis,
        cv2.COLOR_BGR2GRAY
    )


    # =====================
    # 2.
    # 红外去噪
    # =====================

    ir_clean=cv2.bilateralFilter(
        ir,
        7,
        50,
        50
    )


    # =====================
    # 3.
    # 提取细节
    # =====================

    ir_detail=gradient(
        ir_clean
    )

    vis_detail=gradient(
        vis_gray
    )



    # =====================
    # 4.
    # 红外细节选择
    # =====================

    ir_choose=(
        ir_detail >
        vis_detail + detail_threshold
    )



    # =====================
    # 5.
    # 热目标保护
    # =====================


    local=cv2.blur(
        ir_clean,
        (15,15)
    )


    contrast=(
        ir_clean.astype(float)
        -
        local
    )


    thermal_mask=(
        contrast>25
    )



    # 热目标强制加入

    ir_choose = (
        ir_choose |
        thermal_mask
    )



    # =====================
    # 6.
    # 融合
    # =====================


    fusion=np.where(
        ir_choose,
        ir_clean,
        vis_gray
    )


    return fusion.astype(
        np.uint8
    )

ir = cv2.imread("D:/machinelearn/infrared_and_visible_image_fusion/Tar/SDIF-net/datasets/M3FD_Detection/ir/00333.png",cv2.IMREAD_GRAYSCALE)
vis = cv2.imread("D:/machinelearn/infrared_and_visible_image_fusion/Tar/SDIF-net/datasets/M3FD_Detection/vi/00333.png")
fusion = robust_ir_vis_fusion(ir,vis)
cv2.imwrite("fusion.png",fusion)
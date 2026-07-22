import cv2
import numpy as np



def intensity_balance_ir_vis_gray(
        ir,
        vis,
        max_scale=3.0
):
    """
    红外-可见光灰度亮度平衡

    参数:
    ----------
    ir:
        红外图
        shape(H,W)
        uint8


    vis:
        可见光RGB图
        shape(H,W,3)
        uint8


    max_scale:
        最大增强倍数


    返回:
    ----------
    ir_out:
        输出红外图


    vis_gray_out:
        输出灰度可见光图

    """



    # ===============================
    # 1. 可见光 RGB -> 灰度
    # ===============================

    vis_gray = cv2.cvtColor(
        vis,
        cv2.COLOR_BGR2GRAY
    )



    # ===============================
    # 2. 计算平均亮度
    # ===============================

    ir_mean = np.mean(ir)

    vis_mean = np.mean(vis_gray)


    eps = 1e-6



    # ===============================
    # 情况1:
    # 红外暗
    # 增强红外
    # ===============================

    if ir_mean < vis_mean:


        scale = (
            vis_mean /
            (ir_mean + eps)
        )


        # 防止过曝
        scale = min(
            scale,
            max_scale
        )


        ir_out = (
            ir.astype(np.float32)
            * scale
        )


        ir_out = np.clip(
            ir_out,
            0,
            255
        )


        ir_out = (
            ir_out
            .astype(np.uint8)
        )


        # 可见光灰度保持

        vis_gray_out = vis_gray



    # ===============================
    # 情况2:
    # 红外亮
    # 增强可见光灰度
    # ===============================

    else:


        scale = (
            ir_mean /
            (vis_mean + eps)
        )


        scale = min(
            scale,
            max_scale
        )


        vis_gray_out = (
            vis_gray.astype(np.float32)
            * scale
        )


        vis_gray_out = np.clip(
            vis_gray_out,
            0,
            255
        )

        vis_gray_out = (
            vis_gray_out
            .astype(np.uint8)
        )
        # 红外保持

        ir_out = ir

    return ir_out, vis_gray_out

# 读取红外


ir = cv2.imread("D:/machinelearn/infrared_and_visible_image_fusion/Tar/SDIF-net/datasets/M3FD_Detection/ir/00333.png",0)

vis = cv2.imread("D:/machinelearn/infrared_and_visible_image_fusion/Tar/SDIF-net/datasets/M3FD_Detection/vi/00333.png")



ir_new, vis_gray_new = intensity_balance_ir_vis_gray(
    ir,
    vis
)



cv2.imwrite(
    "ir_result00333.png",
    ir_new
)


cv2.imwrite(
    "vis_gray_result00333.png",
    vis_gray_new
)


#-------------------------



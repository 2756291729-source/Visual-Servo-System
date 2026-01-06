import numpy as np
import cv2
import glob
import os

# 1. 定义棋盘格的尺寸
chessboard_size = (11, 8)  # 请根据你的标定板实际内角点数量修改！
square_size = 30.0  # 请修改为你的标定板方格的实际尺寸！

# 2. 准备世界坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 3. 存储坐标点
objpoints = []
imgpoints = []

# 4. 读取标定图片
images_path = glob.glob('photo/*.bmp')  # 请修改为你的图片路径！

# 进一步缩小显示比例
display_scale = 0.3  # 缩小为原始尺寸的30%

for image_path in images_path:
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 5. 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret == True:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_refined)

        # 绘制角点
        img_with_corners = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners_refined, ret)

        # 缩小图像后再显示
        resized_img = cv2.resize(img_with_corners, None, fx=display_scale, fy=display_scale)
        cv2.imshow('Found Corners', resized_img)
        cv2.waitKey(500)
    else:
        print(f"未在图片 {image_path} 中找到角点")

cv2.destroyAllWindows()

# 6. 检查有效图片数量
if len(objpoints) < 10:
    print(f"警告：只找到了 {len(objpoints)} 张有效图片。建议至少10张以上")
else:
    print(f"找到了 {len(objpoints)} 张有效图片，开始标定...")

# 7. 相机标定
if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # 8. 打印结果
    print("\n=== 标定结果 ===")
    print("重投影误差:", ret)
    print("\n相机内参矩阵:\n", camera_matrix)
    print("\n畸变系数:\n", dist_coeffs.ravel())

    # 9. 保存结果
    np.savez('camera_calibration_results.npz',
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)
    print("\n标定参数已保存")

    # 10. 测试校正效果（添加缩放显示）
    test_img = cv2.imread(images_path[0])
    if test_img is not None:
        h, w = test_img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # 进一步缩小显示
        resized_original = cv2.resize(test_img, None, fx=display_scale, fy=display_scale)
        resized_undistorted = cv2.resize(undistorted_img, None, fx=display_scale, fy=display_scale)

        cv2.imshow('Original Image', resized_original)
        cv2.imshow('Undistorted Image', resized_undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("无法读取测试图片")
else:
    print("没有找到有效的标定图片，无法进行标定")
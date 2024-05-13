import cv2
import numpy as np
import os
import random
import imutils

# Hàm thực hiện augmentation nhiễu Gauss và lưu ảnh
def augmentation_and_save(image_path, output_path_prefix, num_augmentations=1):
    # Đọc ảnh gốc
    image = cv2.imread(image_path)

    # Áp dụng augmentation
    for i in range(num_augmentations):
        # Thực hiện augmentation 
        # Thêm nhiễu Gaussian
        mean = np.random.randint(0,100)
        std_dev = np.random.randint(0,100)
        noise = np.zeros(image.shape, np.uint8)
        cv2.randn(noise, mean, std_dev)
        noisy_image = cv2.add(image, noise)

        # Lưu ảnh đã augmentation
        output_path = f"{output_path_prefix}_nhieu_gauss_{i}.jpg"
        cv2.imwrite(output_path, noisy_image)

def add_boder(image_path, output_path, low, high):
    """
    low: kích thước biên thấp nhất (pixel)
    hight: kích thước biên lớn nhất (pixel)
    """
    # random các kích thước biên trong khoảng (low, high)
    top = random.randint(low, high)
    bottom = random.randint(low, high)
    left = random.randint(low, high)
    right = random.randint(low, high)
    
    image = cv2.imread(image_path)
    original_width, original_height = image.shape[1], image.shape[0]
    
    #sử dụng hàm của opencv để thêm biên
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    
    #sau đó resize ảnh bằng kích thước ban đầu của ảnh
    image = cv2.resize(image, (original_width, original_height))
    output_path = f"{output_path}_add_boder.jpg"
    cv2.imwrite(output_path, image)


# Crop ảnh
def random_crop(image_path, out_path):
    image = cv2.imread(image_path)
    
    original_width, original_height = image.shape[1], image.shape[0]
    x_center,y_center = original_height//2, original_width//2
    
    x_left = random.randint(0, x_center//2)
    x_right = random.randint(original_width-x_center//2, original_width)
    
    y_top = random.randint(0, y_center//2)
    y_bottom = random.randint(original_height-y_center//2, original_width)
    
    # crop ra vùng ảnh với kích thước ngẫu nhiên
    cropped_image = image[y_top:y_bottom, x_left:x_right]
    # resize ảnh bằng kích thước ảnh ban đầu 
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    output_path = f"{out_path}_crop.jpg"
    cv2.imwrite(output_path, cropped_image)

#thay đổi độ sáng
def change_brightness(image_path, output_path, value):
    """
    value: độ sáng thay đổi
    """
    img=cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    output_path = f"{output_path}_change_brightness.jpg"
    cv2.imwrite(output_path, img)


def rotate_image(image_path, range_angle, output_path):
    """
    range_angle: Khoảng góc quay
    """
    image = cv2.imread(image_path)
    #lựa chọn ngẫu nhiên góc quay 
    angle = random.randint(-range_angle, range_angle)
    
    img_rot = imutils.rotate(image, angle)
    output_path = f"{output_path}_change_angle.jpg"
    cv2.imwrite(output_path, img_rot)

# Đường dẫn tới thư mục chứa ảnh gốc
input_dir = "LP2"

# Đường dẫn tới thư mục lưu ảnh đã augmentation
output_dir = "augmented_images_LisencePlate/"

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lấy danh sách các file ảnh trong thư mục đầu vào
image_files = os.listdir(input_dir)

# Thực hiện các hàm augmentation cho từng ảnh trong file 
for image_file in image_files:
    input_image_path = os.path.join(input_dir, image_file)
    output_image_prefix = os.path.join(output_dir, os.path.splitext(image_file)[0])
    augmentation_and_save(input_image_path, output_image_prefix)
    add_boder(input_image_path,output_image_prefix,0,500)
    random_crop(input_image_path,output_image_prefix)
    change_brightness(input_image_path,output_image_prefix,100)
    rotate_image(input_image_path,70,output_image_prefix)
    
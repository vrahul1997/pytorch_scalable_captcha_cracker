from PIL import Image
import io

img = Image.open("input/mca_test_files/3487.png", mode='r')
imgByteArr = io.BytesIO()
img.save(imgByteArr, format='PNG')
imgByteArr = imgByteArr.getvalue()
pass
import pandas as pd
import numpy as np
import glob
import os

# file = pd.read_csv("src/captcha.csv")


# print(file["Captcha Value"][:2249])
# image_files = glob.glob(os.path.join())


# def frst_char(x):
#     x = x.split("/")[3].split(".")[0]
#     # print(x)
#     return int(x)


# image_files = sorted(glob.glob("../input/captcha_v2/*.png"), key=frst_char)

# df = pd.read_csv("src/mca_captcha.csv")


file_name = []
for count, filename in enumerate(os.listdir("../input/raw_captcha/")):

    # try:
    #     filenum = str(filename)
    # except Exception as e:
    #     print(e)
    #     filenum = 5000
    # if filenum in df["Image Name"].values:
    #     cap_value = df[df["Image Name"] == filenum]["Captcha Value"].values[0]
    #     print(filenum, cap_value)
    #     src = "input/raw_captcha/" + str(filename)
    #     dst = "input/raw_captcha/" + str(cap_value) + ".png"
    #     os.rename(src, dst)

    # if os.stat("../input/captcha_v2/" + str(filename)).st_size < 3000:
    #     print(filename)
    # else:
    #     pass

    if len(filename.split('.')[0]) == 8:
        # file_name = filename.split('.')[0]
        # src = "../input/raw_captcha/" + str(filename)
        # dst = "../input/raw_captcha/" + str(file_name) + "#" + ".png"
        # os.rename(src, dst)

        print(filename)


#     file_name.append(str(filename.split(".")[0]))

# for index, i in enumerate(df["Captcha Value"][:2249].values):
#     if i in file_name:
#         pass
#     else:
#         pass
#         print(df[df["Captcha Value"] == i]["Image Name"], index)

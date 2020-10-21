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

# df = pd.read_csv("input/mca_captcha/mca_captcha.csv")


# file_name = []
# coun = 0
h = ['2a3rw', '2a4yp', '2aca7', '2adfn', '2akfc', '2b32f', '2b7f6', '2b8d', '2bap8', '2bdnf', '2be36',
     '2bka2', '2bkd8', '2bn6e', '2bncx', '2bndn', '2bwa', '2c3yn', '2c52f', '2c5em', '2c7y5', '2c8r6',
     '2ced5', '2cp47', '2cwre', '2d2da', '2d7na', '2da4m', '2db45', '2db54', '2dfme', '2dr7p', '2drfn',
     '2dx2f', '2dxf3', '2dy68', '2e3n4', '2e7w8', '2e8bm', '2ea7p', '2edy6', '2eg4n', '2em6r', '2ep84',
     '2epk5', '2er73', '2erw2', '2ey3p', '2f7fb', '2fakc', '2fhr8', '2fpe6', '2frh5', '2fwmn', '2g3nm',
     '2g47e', '2g4rk', '2g7fc', '2gy7', '2gpcf', '2gw7w', '2h32d', '2h4rh', '2h6mr', '2h6my', '2h7h6',
     '2hcxc', '2hdk5', '2hew3', '2hfbd', '2hfr', '2hgrb', '2hmgd', '2hnra', '2hwa', '2hwna', '2k242',
     '2k4ak', '2k7ph', '2k8gy', '2kf6p', '2kgpg', '2kxpe', '2ma65', '2max2', '2mb3r', '2mehy', '2mk8k',
     '2mkg', '2m6d', '2max', '2mnrd', '2mpa7', '2mw7r', '2n43x', '2nay3', '2ngyh', '2nhr6', '2np2b', '2np6b',
     '3BEC7', '3BGKY', '3BGS3', '3BHR3T', '3BJNQ', '3BYXW', '3C5DY', '3CE6NQ', '3CH6X', '3CMLK', '3CRTE', '3CT63',
     '3CUFWC', '3CX7RT', '3DCMG', '3DF3C', '3DGQU', '3DKCK', '3DVNEP', '3EDCPN', '3EQLK', '3ERJF', '3ETDF', '3FBEM',
     '3FCMSQ', '3FCPY', '3FETY', '3FHJS', '3FHNH', '3FX3U', '3GCQ5', '3GDL5', '3GE3C3', '3GNR7', '3GWJ6Y', '3H5HN4',
     '3H5RP', '3HFWK', '3HG3Y', '3HQUV', '3HR4F', '3HUNX', '3HWYF', '3J3EN', '3J5KXL', '3J5MJ', '3J7UM', '3JKS6',
     '3JNGM7', '3JPVY', '3K37S', '3K6BT', '3KBG7', '3KRSK', '3KTYN', '3KUFW', '3KUVQ', '3KYQPC', '3KY', '3L3PD',
     '3L6GTX', '3LB7W', '3LCGP', '3LEHT', '3LETC', '3LU7RW', '3LV6G', '3LBT', '3LXHF', '3M3YG', '3M4BW', '3M76Q',
     '3MG6M', '3MHC6', '3MNXFG', '3MPNS7', '3MST5', '3MSXP', '3N4DUX', '3N5VGT', '3NB7K', '3NE3C', '3NEJTF', '3NLG6',
     '3N3R', '3NP7P7', '3NPB3L', '3NWK6C', '3P57P', '3PSP7', '3PSWS', '3PTLQ', '3PUK7', '3QESG', '3QGPB', '3QCT', '3QRQ6', '3QTU7', '3QXET', '3R3WB',
     '3R47', '3R5V', '3RCHL', '3RJYB', '3RPWV', '3RVPX', '3SB3D', '3SFR3', '3SNTB', '3SCG', '3SJL', '3T3CF', '3T5JV', '3T6HR', '3T6RM5',
     '3TBW5M', '3TDYP', '3TLDC', '3TLT6', '3TMHW', '3TPGX', '3TR6W', '3TWN3', '3U3L5', '3U4YL', '3UBQC', '3UDS6', '3UQ63Q',
     '3UXLS', '3V3YR', '3V4JBY', '3V5QL', '3V64B', '3V7JB', '3VBFD', '3VJBU', '3VLM6', '3VLR6B', '3VMS65', '3VRLY', '3VTQL',
     '3VU3J', '3VWTR', '3VXPB', '3W43W', '3WBMS', '3WEQ5', '3WGC3', '3WLBW', '3WPX', '3WQS6', '3WNT', '3X7GD', '3XBYP', '3XELV',
     '3XJ3K6', '3XJ5', '3XLY3', '3XM4F', '3XNYJ7', '3XQU5', '3XSBR', '3Y56V', '3Y6QT', '3YECE', '3YEV6', '3YQSB', '3YT6H', '3YTVX',
     '3YWC6', '3YWR7', '4B4HB', '4B7NJ', '4BKFL', '4BLPG', '4BQSX', '4BRCSM', '4BSB3', '4BVYPG', '4C3FV', '4C64LY', '4C7JCL', '4CDKX',
     '4CKF7', '4CQMRD', '4CR3W', '4CTE3B', '4CUX3', '4CVDY', '4CX65', '4D5FRS', '4D6LGT', '4DCHP', '4DGJ', '4DSMV', '4DYVB', '4EDJF', '4EG65',
     '4F3L', 'atrtas', 'basers', 'beking', 'blotion', 'branter', 'broler', 'crareby', 'carging', 'cauced', 'corsted', 'champer', 'churths',
     'cogrts', 'conts', 'cophtly', 'daners', 'deamer', 'degler', 'dusmng', 'facned', 'faring', 'femers', 'foling', 'forder', 'gramons',
     'dology', 'graener', 'guirest', 'duiters', 'guizing', 'halkel', 'hancler', 'harsts', 'hatning', 'houest', 'hounds', 'humter', 'litstle',
     'incged', 'incking', 'inscky', 'jewing', 'latker', 'liuses', 'lisrded', 'locked', 'lowtest', 'maders', 'uantes', 'motged', 'motked', 'movoth',
     'ieings', 'migings', 'ownly', 'pagand', 'paging', 'pardit', 'parlar', 'piper', 'plehes', 'plourer', 'prihed', 'radok', 'redcing', 'sanolds',
     'shaely', 'shases', 'skiser', 'snening', 'sofsted', 'smohin', 'spoche', 'stolers', 'strled', 'talets', 'thithm', 'tineter', 'tlined', 'traerts',
     'tursent', 'uncter', 'unhlish', 'unlreas', 'unslen', 'unster', 'vling', 'verdle', 'vieles', 'walcers', 'walrty', 'wavuid', 'wilker', 'wirios',
     'witying', 'wolver', 'wond', 'worusts', 'uvrters', 'yelmed']

print(len(h))

df = pd.read_csv("../input/test_values.csv")
df['pred_values'] = h
df.to_csv("../input/test_values.csv", index=False)

print(len(df[df["file_name"] == df["pred_values"]]["file_name"]))

# for count, filename in enumerate(os.listdir("../input/all_test/")):

# filename = str(filename.split(".")[0])

#     h.append(filename)
# j = sorted(h)

# df = pd.DataFrame(columns=['file_name'], data=j)
# print(df.file_name)
# print([i for i in j if j.count(i) > 0])

# df.to_csv("../input/test_values.csv", index=False)

#     if filenum in df["Image Name"].values:
#         cap_value = df[df["Image Name"] == filenum]["Captcha Value"].values[0]
#         print(filenum, cap_value)
#         src = "input/mca_captcha/test_images/" + str(filename)
#         dst = "input/mca_captcha/test_images/" + str(cap_value) + ".png"
#         os.rename(src, dst)

# if os.stat("../input/captcha_v2/" + str(filename)).st_size < 3000:
#     print(filename)
# else:
#     pass

#     if len(filename) == 7:
#         # # file_name = filename.split('.')[0]
#         # src = "../input/all_captchas/" + str(filename) + ".png"
#         # dst = "../input/all_captchas/" + str(filename) + "#" + ".png"
#         # os.rename(src, dst)
#         print(filename)
#         coun += 1
# print(coun)

# os.remove("../input/all_captchas/" + filename + ".png")

#     file_name.append(str(filename.split(".")[0]))

# for index, i in enumerate(df["Captcha Value"][:2249].values):
#     if i in file_name:
#         pass
#     else:
#         pass
#         print(df[df["Captcha Value"] == i]["Image Name"], index)

# import glob

# unfiles = glob.glob("../input/dl_captcha/*.txt")
# for i in unfiles:
#     os.remove(i)

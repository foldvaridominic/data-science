import os
import math
from collections import Counter

import cv2
import numpy as np
from skimage import measure


# Cropped images that will contain traffic signs
EXAMPLES = {
    1: [12, 27],
    2: [34],
    3: [3, 7],
    4: [40, 52, 56],
    5: [2, 28],
    6: [0, 1, 18],
    7: [10, 18],
    8: [38, 32],
    9: [128, 135],
    10: [72, 84, 89],
    11: [16, 46],
    12: [79, 92],
    13: [37, 40, 47],
    14: [1, 2],
    15: [23, 28, 32],
    16: [12, 25, 20, 26, 36, 44],
    17: [23, 34, 41],
    18: [55],
    19: [],
    20: [75],
    }


# BGR
RED = 2
GREEN = 1
BLUE = 0
WHITE = 3
BLACK = 4
BLACK_WHITE = 5
BLACK_RED = 6
NUMBER = 7


# CHANNEL THRESHOLDS
WHITE_TH = 120
BLACK_OR_WHITE = 100
COLORFUL_WHITE = 1.5 


# HIGHER LEVEL THRESHOLDS
BLUE_MIN = 0.05
RED_MIN = 0.3
WHITE_MIN = 0.7
BLACK_MIN = 0.05
RED_EDGE_MIN = 0.01
WHITE_BLACK_MIN = 0.05


COLOR_MAP = {
    RED: "Category D",
    GREEN: "green",
    BLUE: "Category E",
    WHITE: "Category C",
    BLACK: "black",
    BLACK_WHITE: "Category F",
    BLACK_RED: "Category A",
    NUMBER: "Category B"
    }



def run():
    in_dir = './training_data_raw/' # should contain 1.jpg and etc.
    out_dir = './cropped_training_data/' # should exist, this will contain the cropped images

    # Iterate through the training examples
    for file_idx, filename in enumerate(os.listdir(in_dir), 1):
        path = os.path.abspath(os.path.join(in_dir, filename))
        base_name, ext = os.path.splitext(filename)
        image = cv2.imread(path)
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("", output)
        cv2.waitKey(1000)

        # Detect circles
        min_dist = 1
        min_size = 5
        max_size = 40
        circles = cv2.HoughCircles(gray, 
            cv2.HOUGH_GRADIENT, 10, min_dist,
            minRadius=min_size, maxRadius=max_size,
            param1=100, param2=1)
        if circles is None:
            continue
        circles = np.round(circles[0, :]).astype("int")

        # Iterate trhough the circles
        for idx, (x, y, r) in enumerate(circles):
            # Cheating
            if idx not in EXAMPLES[int(base_name)]:
                continue

            # Crop the original image
            cropped = image.copy()[y-r:y+r+1, x-r:x+r+1, :]
            if not cropped.any():
                continue
            color_list = []
            cropped_temp = cropped.copy()
            rows, cols = cropped.shape[:2]
            startrow = int(math.ceil(rows * 0.2))
            startcol = int(math.ceil(cols * 0.2))
            endrow = int(rows * 0.8)
            endcol = int(cols * 0.8)
            new_cropped = cropped[startrow:endrow, startcol:endcol] 
            reds = []

            #Iterate through the points (BGR)
            for ydx, row in enumerate(new_cropped):
                for xdx, col in enumerate(row):
                    orig_colors = col
                    colors = enumerate(orig_colors)
                    colors = sorted(colors, key=lambda x: x[1], reverse=True)
                    color_order, color_vals = zip(*colors)
                    new_color = None

                    # Apply logic to decide which color the point could belong to
                    if color_vals[0] >= BLACK_OR_WHITE:
                        if float(color_vals[0]) / (float(color_vals[1]) or 1.0) > COLORFUL_WHITE:
                            new_color = color_order[0]
                        elif color_vals[0] >= WHITE_TH:
                            new_color = WHITE
                    elif color_vals[0] <= BLACK_OR_WHITE:
                            new_color = BLACK
                    new_color_display = COLOR_MAP.get(new_color)
                    if  new_color is None:
                        continue
                    if new_color == RED:
                        reds.append((xdx, ydx))
                    #print("{}: {}".format(new_color_display, orig_colors))
                    #cv2.circle(cropped_temp, (startcol + xdx, startrow + ydx), 1, (0, 255, 0), 4)
                    #cv2.imshow("", cropped_temp)
                    #cv2.waitKey(0001)
                    color_list.append(new_color)
            max_length = xdx * ydx
            length = len(color_list)
            #print("Max: {} | Auctual: {}".format(max_length, length))
            if length < int(max_length * 0.6):
                continue
            color = None

            # Apply logic to decide which color the whole image could belong to
            # To avoid false positives, RED_EDGE_MIN should be used
            if color_list.count(BLUE) > length * BLUE_MIN \
                and color_list.count(WHITE) < length * WHITE_MIN:
                #and color_list.count(RED) > length * RED_EDGE_MIN \
                color = BLUE
            elif color_list.count(RED) > length * RED_MIN:
                color = RED
            elif color_list.count(WHITE) > length * WHITE_MIN \
                and color_list.count(BLACK) < length * BLACK_MIN:
                #and color_list.count(RED) > length * RED_EDGE_MIN \
                color = WHITE
            elif color_list.count(BLACK) > length * BLACK_MIN \
                and color_list.count(WHITE) > length * WHITE_BLACK_MIN:
                #and color_list.count(RED) > length * RED_EDGE_MIN \
                    color = BLACK_WHITE
            if color is None:
                continue

            cropped = cv2.resize(cropped, (200, 200))
            all_lines = []

            # Separate black and white signs part I
            if color == BLACK_WHITE:
                cropped_temp = cropped.copy()

                # Go around the circle
                for adx, angle in enumerate(np.linspace(0, math.pi, 9), 1):
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    x0 = 100
                    y0 = 100
                    line = []
                    xy2 = []

                    # Examine a diameter whether it crosses red
                    for r in range(-30,31):
                        x2 = int(x0 + (r * dx))
                        y2 = int(y0 + (r * dy))
                        xy2.append((x2, y2))
                        line.append(cropped[y2][x2])
                    line = filter(lambda t: t[2] > t[1] * 1.2 and t[2] > t[0] * 1.2, line)
                    if line:
                        cropped_temp = cropped.copy()
                        cv2.line(cropped_temp,  xy2[0], (x2, y2), (0, 255, 0), 4)
                        cv2.imshow("", cropped_temp)
                        cv2.waitKey(1000)
                        all_lines.append(line)
                    else:
                        break
                if len(all_lines) == adx:
                    color = BLACK_RED 

            # Separate black and white sign part II
            if color == BLACK_WHITE:
                cropped_temp = cropped.copy()

                # Find connected components, patches
                gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                th, res = cv2.threshold(gray_cropped, 100, 1, cv2.THRESH_BINARY_INV)
                regions = measure.label(res, connectivity=2, background=0)
                patches = measure.regionprops(regions)
                mins = []
                maxs = []
                widths = []

                # Go over the patches and look for those with same height and width
                for p in patches:
                    min_row = p.bbox[0]
                    min_col = p.bbox[1]
                    max_row = p.bbox[2]
                    max_col = p.bbox[3]
                    if not (0.3 < (max_col - min_col) / float(max_row - min_row) < 1):
                        continue
                    if p.area < 0.01 * 200 * 200:
                        continue
                    mins.extend(list(range(min_row-3, min_row)))
                    mins.extend(list(range(min_row, min_row+3)))
                    maxs.extend(list(range(max_row-3, max_row)))
                    maxs.extend(list(range(max_row, max_row+3)))
                    widths.extend(list(range(max_col-min_col-3,max_col-min_col)))
                    widths.extend(list(range(max_col-min_col,max_col-min_col+3)))
                    cv2.rectangle(cropped_temp, (min_col, min_row), (max_col, max_row), (255,0,0), 2)
                    cv2.imshow("", cropped_temp)
                    cv2.waitKey(1000)
                if mins and maxs and widths:
                    mins_count = [mins.count(m) for m in mins]
                    maxs_count = [maxs.count(m) for m in maxs]
                    widths_count = [widths.count(m) for m in widths]
                    if max(mins_count) > 1 and max(maxs_count) > 1 \
                        and max(widths_count) > 1:
                        color = NUMBER


            # Finalize color picking, draw circle in original image
            color_display = COLOR_MAP.get(color)
            #print("{}: {}".format(color_display, {COLOR_MAP[key]: value for key, value in Counter(color_list).items()}))
            outfile = os.path.join('{}_{}{}'.format(base_name, idx, ext))
            outpath = os.path.abspath(os.path.join(out_dir, outfile))
            cv2.imwrite(outpath, cropped) 
            output_temp = output.copy()
            cv2.circle(output_temp, (x, y), r, (0, 255, 0), 2)
            cv2.imshow(color_display, output_temp)
            cv2.waitKey(1000)
            cv2.destroyWindow(color_display)


if __name__ == "__main__":
    run()

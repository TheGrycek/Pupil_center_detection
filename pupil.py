import math as m
import random

import cv2 as cv
import numpy as np


class Pupil:
    def __init__(self, img_path, write_path):
        self.img_path = img_path
        self.image = cv.imread(img_path, 0)
        self.write_path = write_path

    def parallelogram(self):
        x_l_old, y_u_old = 1000, 1000
        x_r_old, y_d_old = 0, 0
        x_k, y_k = 0, 0
        cx, cy = 0, 0
        a = 0

        # THRESHOLDING
        img = self.image
        ret, thresh = cv.threshold(img, 40, 255, cv.THRESH_BINARY)

        # CENTER APPROXIMATION
        conturs, hierarhy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(conturs) > 1 is not None:
            M = cv.moments(conturs[1])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            np.float32([cx, cy])

            # # INTEREST REGION
            frame = thresh[cy - 50: cy + 50, cx - 50: cx + 50]

            # # FILTERING
            frame = cv.medianBlur(frame, 3)
            size = frame.shape

            for y in range(0, size[0]):
                for x in range(0, size[1]):
                    if frame[y, x] == 0:
                        if x < x_l_old:
                            x_l_old = x
                            
                        if x > x_r_old:
                            x_r_old = x

                frame[y, x_l_old: x_r_old] = 0
                x_l_old = 1000
                x_r_old = 0

            for x in range(0, size[1]):
                for y in range(0, size[0]):
                    if frame[y, x] == 0:
                        if y < y_u_old:
                            y_u_old = y

                        if y > y_d_old:
                            y_d_old = y

                frame[y_u_old: y_d_old, x] = 0
                y_u_old = 1000
                y_d_old = 0

            # # PUPIL EDGES
            img_edges = cv.Canny(frame, 100, 255)
            img_edges = abs(255 - img_edges)

            # # CENTER DETECTION - PARALLELOGRAM METHOD
            self.image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
            x_k, y_k = self.pupil_center(img_edges, self.image, cx - 50, cy - 50)

            if x_k != 0 and y_k != 0:
                a, b, fangle = self.calculate_pupil_diameter(img_edges, self.image, x_k, y_k, cx - 50, cy - 50)

        if x_k and y_k and a is not None:
            return int(x_k + cx - 50), int(y_k + cy - 50), a, b, fangle
        else:
            return int(0), int(0), int(0)

    @staticmethod
    def gauss(m):
        # eliminate columns
        for col in range(len(m[0])):
            for row in range(col + 1, len(m)):
                if m[col][col] != 0:
                    r = [(rowValue * (-(m[row][col] / m[col][col]))) for rowValue in m[col]]
                    m[row] = [sum(pair) for pair in zip(m[row], r)]

        ans = []
        m.reverse()
        for sol in range(len(m)):
            if sol == 0:
                ans.append(m[sol][-1] / m[sol][-2] if m[sol][-2] != 0 else 0)

            else:
                inner = 0
                for x in range(sol):
                    inner += (ans[x] * m[sol][-2 - x])
                ans.append((m[sol][-1] - inner) / m[sol][-sol - 2] if m[sol][-sol - 2] != 0 else 0)

        ans.reverse()
        if ans is not None:
            return ans
        else:
            return 0

    def calculate_pupil_diameter(self, img, img_main, cx, cy, cx_shift, cy_shift):
        line_size, point_size = 2, 2
        img_edges = img
        points = []

        # FINDING POINTS ON ELIPSE
        for y in range(img_edges.shape[0]):
            x1, x2 = 0, 0
            for j in range(img_edges.shape[1]):
                if (img_edges[y][j]) <= 200:
                    x1 = j
                    break
            for k in range(img_edges.shape[1] - 1, 0, -1):
                if (img_edges[y][k]) <= 200:
                    x2 = k
                    points.append([y, x1, x2])
                    break

        max_error = 0.0001
        error_x, error_y = 1, 1
        iterator = 0

        # FINDING BEST FACTORS
        while not (error_y < max_error and error_x < max_error):
            # DRAWING POINTS ON BOTTOM PART OF ELLIPSE
            result = []
            result.append(points[random.randint(int(len(points) * 0.5), int(len(points) * 0.75))])
            result.append(points[random.randint(int(len(points) * 0.75), int(len(points) * 0.95))])
            result.append(points[len(points) - 1])

            x1, x2, x3 = (result[2][1] + result[2][2]) / 2, result[1][1], result[1][2]
            y1, y2 = result[2][0], result[1][0]
            y3 = y2
            x4, x5 = result[0][1], result[0][2]
            y4 = result[0][0]
            y5 = y4

            # CREATING A MATRIX
            A_matrix = [[x1 * x1, x1 * y1, y1 * y1, x1, y1, -1],
                        [x2 * x2, x2 * y2, y2 * y2, x2, y2, -1],
                        [x3 * x3, x3 * y3, y3 * y3, x3, y3, -1],
                        [x4 * x4, x4 * y4, y4 * y4, x4, y4, -1],
                        [x5 * x5, x5 * y5, y5 * y5, x5, y5, -1]]

            # CALCULATING FACTORS OF ELIPSE EQUATION
            factors = self.gauss(A_matrix)

            A, B, C, D, E = factors
            F = -((A * x1 * x1) + (B * x1 * y1) + (C * y1 * y1) + (D * x1) + (E * y1))

            # CALCULATING ELIPSE CENTRAL POINT
            x0 = int(((B * E) - (2 * C * D)) / ((4 * A * C) - (B * B)))
            y0 = int((B * D - 2 * A * E) / (4 * A * C - B * B))

            # CHECKING PREVIOUS RESULTS
            if cx > 0 and cy > 0:
                error_x = abs(x0 - (cx)) / cx
                error_y = abs(y0 - (cy)) / cy

            iterator += 1

        a = ((2 * (A * x0 * x0 + B * x0 * y0 + C * y0 * y0 - F)) / ((A + C) -
                                                                    (B * B + (A - C) * (A - C)) ** 0.5)) ** 0.5
        b = ((2 * (A * x0 * x0 + B * x0 * y0 + C * y0 * y0 - F)) / ((A + C) +
                                                                    (B * B + (A - C) * (A - C)) ** 0.5)) ** 0.5
        fangle = 0.5 * a * m.atan(B / (A - C))

        print("Results: a=", a, "b=", b, "fangle=", fangle)

        # DRAWING POINTS
        img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2BGR)

        res_indices = [[2, 1, 2, 0], [1, 1, 1, 0], [1, 2, 1, 0], [0, 1, 0, 0], [0, 2, 0, 0]]
        for i, j, k, l in res_indices:
            cv.circle(img_edges, (result[i][j], result[k][l]), 2, (0, 127, 17), line_size)

        # DRAWING CENTRAL POINTS
        cv.line(img_edges, (cx - 5, cy), (cx + 5, cy), (0, 0, 255), point_size)
        cv.line(img_edges, (cx, cy - 5), (cx, cy + 5), (0, 0, 255), point_size)

        # DRAWING IMAGE CENTER
        cv.line(img_edges, (int(img_edges.shape[0] / 2), 0),
                (int(img_edges.shape[0] / 2), img_edges.shape[0]), (0, 255, 0), 1)
        cv.line(img_edges, (0, int(img_edges.shape[1] / 2)),
                (int(img_edges.shape[0]), int(img_edges.shape[1] / 2)), (0, 255, 0), 1)

        # DRAWING ELIPSE
        img_mask = np.zeros_like(img_main)
        cv.ellipse(img_mask, (x0 + cx_shift, y0 + cy_shift),
                   (int(a) + 1, int(b) + 5), int(fangle), 0, 360, (255, 255, 255), 1)

        cv.ellipse(img_main, (x0 + cx_shift, y0 + cy_shift),
                   (int(a) + 1, int(b) + 5), int(fangle), 0, 360, (0, 200, 255), 1)
        # EDGES
        cv.ellipse(img_edges, (x0, y0), (int(a) + 2, int(b) + 5), int(fangle), 0, 360, (0, 200, 255), 2)

        if len(img_mask.shape) == 3:
            img_mask = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)

        conturs, hierarhy = cv.findContours(img_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img_mask, conturs, 0, (255, 255, 255), 1)

        # cv.imshow('Ellipse drawing', cv.resize(img_mask, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC))
        # cv.imshow('Ellipse drawing2', cv.resize(img_main, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC))
        cv.imwrite(self.write_path, img_main)

        if a and b and fangle is not None:
            return a, b, fangle
        else:
            return int(0), int(0), int(0)

    @staticmethod
    def pupil_center(img, img_main, cx_shift, cy_shift):
        # CALCULATING PUPIL CENTER
        line_size = 2
        img_edges = img.copy()
        points = []

        for y in range(img_edges.shape[0]):
            x1, x2 = 0, 0
            for j in range(img_edges.shape[1]):
                if (img_edges[y][j]) <= 200:
                    x1 = j
                    break
            for k in range(img_edges.shape[1] - 1, 0, -1):
                if (img_edges[y][k]) <= 200:
                    x2 = k
                    points.append([y, x1, x2, k - j])
                    break

        results = []

        for i in range(int(len(points) / 2)):
            for j in range(len(points) - 1, int(len(points) / 2), -1):
                if points[i][3] == points[j][3]:
                    results.append([points[i], points[j]])
                    break

        if len(img_edges.shape) < 3:
            img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2BGR)

        j = 0
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
        center_points = []

        res_indices = [[0, 1, 0, 0, 0, 2, 0, 0],
                       [1, 1, 1, 0, 1, 2, 1, 0],
                       [0, 1, 0, 0, 1, 1, 1, 0],
                       [1, 2, 1, 0, 0, 2, 0, 0]]

        if int(np.round(len(results) / 3)) > 0:
            for i in range(0, len(results), int(np.round(len(results) / 3))):
                for a, b, c, d, e, f, g, h in res_indices:
                    cv.line(img_edges, (results[i][a][b], results[i][c][d]), (results[i][e][f], results[i][g][h]),
                            color[j], line_size)

                cx1 = int((results[i][0][1] + results[i][1][2]) / 2)
                cy1 = int((results[i][0][0] + results[i][1][0]) / 2)
                center_points.append([cx1, cy1])
                j += 1

            mask = np.ones((img_edges.shape[0], img_edges.shape[1], 3), np.uint8) * 255

            pts = np.array([[center_points[0][0], center_points[0][1]],
                            [center_points[1][0], center_points[1][1]],
                            [center_points[2][0], center_points[2][1]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(mask, [pts], True, (0, 0, 0))

            mask2 = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            conturs, hierarhy = cv.findContours(mask2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            M = cv.moments(conturs[1])
            cx2 = int(M['m10'] / M['m00'])
            cy2 = int(M['m01'] / M['m00'])

        else:
            cx2, cy2 = 0, 0

        if cx2 and cy2 is not None:
            cv.circle(img_main, (cx2 + cx_shift, cy2 + cy_shift), 2, (0, 0, 255), 2)
            # CENTER
            # cv.imshow('Center-Parallelograms Method',
            # cv.resize(img_main, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC))
            return cx2, cy2
        else:
            return int(0), int(0)

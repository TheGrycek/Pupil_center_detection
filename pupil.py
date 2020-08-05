import numpy as np
import cv2 as cv
import random
import math as m

class Pupil:
    def __init__(self, img_path, write_path):
        self.img_path = img_path
        self.image = cv.imread(img_path, 0)
        self.write_path = write_path

    def parallelogram(self):
        # TODO: MAIN LOOP
        x_l_old = 1000
        y_u_old = 1000
        x_r_old = 0
        y_d_old = 0
        x_k = 0
        y_k = 0
        cx = 0
        cy = 0
        a = 0

        # TODO 1: THRESHOLDING
        img = self.image
        ret, thresh = cv.threshold(img, 40, 255, cv.THRESH_BINARY)
        # cv.imshow('oko_progowane.jpg', thresh)

        # TODO 2: CENTER APPROXIMATION
        conturs, hierarhy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(conturs) > 1 is not None:
            M = cv.moments(conturs[1])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            np.float32([cx, cy])

            # # TODO 3: INTEREST REGION
            frame = thresh[cy - 50: cy + 50, cx - 50: cx + 50]

            # # TODO 4: FILTERING
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

            # # TODO 5: PUPIL EDGES
            img_edges = cv.Canny(frame, 100, 255)
            img_edges = abs(255 - img_edges)
            # cv.imshow('oko_kontur.jpg', img_edges)

            # TODO 6: CENTER DETECTION - PARALLELOGRAM METHOD
            self.image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
            x_k, y_k = self.PupilCenter(img_edges, self.image, cx - 50, cy - 50)
            #cv.circle(img, (x_k, y_k), 5, (0, 255, 0), -1)

            if x_k != 0 and y_k != 0:
                a, b, fangle = self.CalculationPupilDiameter(img_edges, self.image, x_k, y_k, cx - 50, cy - 50)
                # a = int(np.round(a))

        if x_k and y_k and a is not None:
            return (int(x_k + cx - 50), int(y_k + cy - 50), a, b, fangle)
        else:
            # return(int(cx), int(cy), int(40))
            return (int(0), int(0), int(0))

    def myGauss(self, m):
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
                if m[sol][-2] != 0:
                    ans.append(m[sol][-1] / m[sol][-2])
                else:
                    ans.append(0)
            else:
                inner = 0
                for x in range(sol):
                    inner += (ans[x] * m[sol][-2 - x])
                if m[sol][-sol - 2] != 0:
                    ans.append((m[sol][-1] - inner) / m[sol][-sol - 2])
                else:
                    ans.append(0)
        ans.reverse()
        if ans is not None:
            return ans
        else:
            return (0)


    def CalculationPupilDiameter(self, img, img_main, cx, cy, cx_shift, cy_shift):

        line_size = 2
        point_size = 2
        img_edges = img
        points = []

        # TODO: FINDING POINTS ON ELIPSE
        for y in range(img_edges.shape[0]):
            x1 = 0
            x2 = 0
            for j in range(img_edges.shape[1]):
                if (img_edges[y][j]) <= 200:
                    # print(i, j)
                    x1 = j
                    break
            for k in range(img_edges.shape[1] - 1, 0, -1):
                if (img_edges[y][k]) <= 200:
                    # print(i, k)
                    x2 = k
                    points.append([y, x1, x2])
                    break

        Max_blad = 0.0001
        bladX = 1
        bladY = 1
        iteracje = 0

        # TODO: FINDING BEST FACTORS
        while (not (bladY < Max_blad and bladX < Max_blad)):

            # TODO: DRAWING POINTS ON BOTTOM PART OF ELIPSE

            wynik = []
            wynik.append(points[random.randint(int(len(points) * 0.5), int(len(points) * 0.75))])
            wynik.append(points[random.randint(int(len(points) * 0.75), int(len(points) * 0.95))])
            wynik.append(points[len(points) - 1])

            x1 = (wynik[2][1] + wynik[2][2]) / 2
            y1 = wynik[2][0]
            x2 = wynik[1][1]
            y2 = wynik[1][0]
            x3 = wynik[1][2]
            y3 = y2
            x4 = wynik[0][1]
            y4 = wynik[0][0]
            x5 = wynik[0][2]
            y5 = y4
            # TODO: CREATING A MATRIX
            Macierz_A = [[x1 * x1, x1 * y1, y1 * y1, x1, y1, -1],
                         [x2 * x2, x2 * y2, y2 * y2, x2, y2, -1],
                         [x3 * x3, x3 * y3, y3 * y3, x3, y3, -1],
                         [x4 * x4, x4 * y4, y4 * y4, x4, y4, -1],
                         [x5 * x5, x5 * y5, y5 * y5, x5, y5, -1]]
            # print(myGauss(Macierz_A))

            # TODO: CALCULATING FACTORS OF ELIPSE EQUATION
            Wspolczyniki = self.myGauss(Macierz_A)

            A = Wspolczyniki[0]
            B = Wspolczyniki[1]
            C = Wspolczyniki[2]
            D = Wspolczyniki[3]
            E = Wspolczyniki[4]
            F = -((A * x1 * x1) + (B * x1 * y1) + (C * y1 * y1) + (D * x1) + (E * y1))

            # TODO: CALCULATING ELIPSE CENTRAL POINT
            x0 = int(((B * E) - (2 * C * D)) / ((4 * A * C) - (B * B)))
            y0 = int((B * D - 2 * A * E) / (4 * A * C - B * B))

            # TODO: CHECKING PREVIOUS RESULTS

            if (cx > 0 and cy > 0):
                bladX = abs(x0 - (cx)) / cx
                bladY = abs(y0 - (cy)) / cy

            iteracje += 1

        # print("Wyniki: x0=", x0, 'y0=', y0, "   Liczba losowan pieciu punktow", iteracje)

        a = ((2 * (A * x0 * x0 + B * x0 * y0 + C * y0 * y0 - F)) / ((A + C) - ((B * B + (A - C) * (A - C))) ** 0.5)) ** 0.5
        b = ((2 * (A * x0 * x0 + B * x0 * y0 + C * y0 * y0 - F)) / ((A + C) + ((B * B + (A - C) * (A - C))) ** 0.5)) ** 0.5
        fangle = 0.5 * a * m.atan(B / (A - C))

        print("Wyniki: a=", a, "b=", b, "fangle=", fangle)

        # TODO: DRAWING POINTS
        img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2BGR)
        cv.circle(img_edges, (wynik[2][1], wynik[2][0]), 2, (0, 127, 17), line_size)
        cv.circle(img_edges, (wynik[1][1], wynik[1][0]), 2, (0, 127, 17), line_size)
        cv.circle(img_edges, (wynik[1][2], wynik[1][0]), 2, (0, 127, 17), line_size)
        cv.circle(img_edges, (wynik[0][1], wynik[0][0]), 2, (0, 127, 17), line_size)
        cv.circle(img_edges, (wynik[0][2], wynik[0][0]), 2, (0, 127, 17), line_size)

        # TODO: DRAWING CENTRAL POINTS

        cv.line(img_edges, (cx - 5, cy), (cx + 5, cy), (0, 0, 255), point_size)
        cv.line(img_edges, (cx, cy - 5), (cx, cy + 5), (0, 0, 255), point_size)

        # TODO: DRAWING IMAGE CENTER

        cv.line(img_edges, (int(img_edges.shape[0] / 2), 0), (int(img_edges.shape[0] / 2), img_edges.shape[0]), (0, 255, 0),
                1)
        cv.line(img_edges, (0, int(img_edges.shape[1] / 2)), (int(img_edges.shape[0]), int(img_edges.shape[1] / 2)),
                (0, 255, 0), 1)

        # TODO: DRAWING ELIPSE
        img_mask = np.zeros_like(img_main)
        cv.ellipse(img_mask, (x0 + cx_shift, y0 + cy_shift), (int(a) + 1, int(b) + 5), int(fangle), 0, 360, (255, 255, 255),
                   1)

        cv.ellipse(img_main, (x0 + cx_shift, y0 + cy_shift), (int(a) + 1, int(b) + 5), int(fangle), 0, 360, (0, 200, 255), 1)
        # EDGES
        cv.ellipse(img_edges, (x0, y0), (int(a) + 2, int(b) + 5), int(fangle), 0, 360, (0, 200, 255), 2)
        # cv.imshow('Algorytm_dzialanie', img_edges)

        if (len(img_mask.shape) == 3):
            img_mask = cv.cvtColor(img_mask, cv.COLOR_BGR2GRAY)
        conturs, hierarhy = cv.findContours(img_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img_mask, conturs, 0, (255, 255, 255), 1)
        # cv.imshow('Rysowanie elipsy', cv.resize(img_mask, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC))
        # cv.imshow('Rysowanie elipsy2', cv.resize(img_main, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC))
        cv.imwrite(self.write_path, img_main)
        if a and b and fangle is not None:
            return (a, b, fangle)
        else:
            return (int(0), int(0), int(0))


    def PupilCenter(self, img, img_main, cx_shift, cy_shift):
        # TODO 5: CALCULATING PUPIL CENTER

        line_size = 2
        img_edges = img.copy()
        points = []

        for y in range(img_edges.shape[0]):
            x1 = 0
            x2 = 0
            for j in range(img_edges.shape[1]):
                if (img_edges[y][j]) <= 200:
                    x1 = j
                    break
            for k in range(img_edges.shape[1] - 1, 0, -1):
                if (img_edges[y][k]) <= 200:
                    x2 = k
                    points.append([y, x1, x2, k - j])
                    break

        wyniki = []

        for i in range(int(len(points) / 2)):
            for j in range(len(points) - 1, int(len(points) / 2), -1):
                if (points[i][3] == points[j][3]):
                    wyniki.append([points[i], points[j]])
                    break
        if (len(img_edges.shape) < 3):
            img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2BGR)

        j = 0

        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]
        CenterPoints = []

        if int(np.round(len(wyniki) / 3)) > 0:
            for i in range(0, len(wyniki), int(np.round(len(wyniki) / 3))):
                cv.line(img_edges, (wyniki[i][0][1], wyniki[i][0][0]), (wyniki[i][0][2], wyniki[i][0][0]), color[j], line_size)
                cv.line(img_edges, (wyniki[i][1][1], wyniki[i][1][0]), (wyniki[i][1][2], wyniki[i][1][0]), color[j], line_size)
                cv.line(img_edges, (wyniki[i][0][1], wyniki[i][0][0]), (wyniki[i][1][1], wyniki[i][1][0]), color[j], line_size)
                cv.line(img_edges, (wyniki[i][1][2], wyniki[i][1][0]), (wyniki[i][0][2], wyniki[i][0][0]), color[j], line_size)
                # # cv.line(img_edges, (X1,Y), (X2,Y),(255,0,0),1)
                cx1 = int((wyniki[i][0][1] + wyniki[i][1][2]) / 2)
                cy1 = int((wyniki[i][0][0] + wyniki[i][1][0]) / 2)
                # print(cx1,cy1)
                # cv.circle(img_edges, (cx1, cy1), 2, color[j], line_size)
                j += 1
                CenterPoints.append([cx1, cy1])

            mask = np.ones((img_edges.shape[0], img_edges.shape[1], 3), np.uint8) * 255

            pts = np.array([[CenterPoints[0][0], CenterPoints[0][1]], [CenterPoints[1][0], CenterPoints[1][1]],
                            [CenterPoints[2][0], CenterPoints[2][1]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv.polylines(mask, [pts], True, (0, 0, 0))

            mask2 = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            conturs, hierarhy = cv.findContours(mask2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # cv.drawContours(mask, conturs[1], 0, (0,255,0), 1)
            M = cv.moments(conturs[1])
            cx2 = int(M['m10'] / M['m00'])
            cy2 = int(M['m01'] / M['m00'])

        else:
            cx2 = 0
            cy2 = 0

        if cx2 and cy2 is not None:

            cv.circle(img_main, (cx2 + cx_shift, cy2 + cy_shift), 2, (0, 0, 255), 2)
            # CENTER
            # cv.imshow('Srodek-Metoda Rownoleglobokow', cv.resize(img_main, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC))
            return (cx2, cy2)
        else:
            return (int(0), int(0))

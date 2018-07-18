
import random
import numpy
import math


class Datasets():
    def circle(x, y, args):
        def getCircleLabel(x, y):
            condition = x ** 2 + y ** 2
            if(condition < 4):
                return 1
            elif(condition > 9 and condition < 16):
                return -1

            return False

        label = getCircleLabel(x, y)
        if(label is False):
            return False
        elif(args[4] is "test"):
            return [x, y, label]

        noiseX = random.uniform(-4, 4) * args[2]
        noiseY = random.uniform(-4, 4) * args[2]
        x += noiseX
        y += noiseY
        if(getCircleLabel(x, y) is not False):
                return [x, y, label]

        return False

    def line(x, y, args):
        def getLineLabel(x, y):
            condition = [-x, -y, x - y]
            condition = condition[args[0]]
            if(condition > -4 and condition < -2):
                return 1
            elif(condition > 2 and condition < 4):
                return -1
            return False

        label = getLineLabel(x, y)
        if(label is False):
            return False
        elif(args[4] is "test"):
            return [x, y, label]

        x += random.uniform(-6, 6) * args[2]
        y += random.uniform(-6, 6) * args[2]

        return [x, y, label]

    def and_(x, y, args):
        def getAndLabel(x, y):
            if(x > 5 or y > 5 or x < -5 or y < -5):
                return False

            condition = [
                (x > 1 and y > 1),
                (x < -1 and y > 1),
                (x > 1 and y < -1),
                (x < -1 and y < -1)]

            for i in range(len(condition)):
                if(i is args[1]):
                    if(condition[i]):
                        return 1
                else:
                    if(condition[i]):
                        return -1
            return False

        label = getAndLabel(x, y)
        if(label is False):
            return False
        elif(args[4] is "test"):
            return [x, y, label]

        x += random.uniform(-6, 6) * args[2]
        y += random.uniform(-6, 6) * args[2]
        if(getAndLabel(x, y) is not False):
            return [x, y, label]

        return False

    def xor(x, y, args):
        def getXORLabel(x, y):
            if(x > 5 or y > 5 or x < -5 or y < -5):
                return False

            if((x > 1 and y > 1) or (x < -1 and y < -1)):
                return 1
            elif((x < -1 and y > 1) or (x > 1 and y < -1)):
                return -1

            return False

        label = getXORLabel(x, y)
        if(label is False):
            return False
        elif(args[4] is "test"):
            return [x, y, label]

        x += random.uniform(-6, 6) * args[2]
        y += random.uniform(-6, 6) * args[2]
        if(getXORLabel(x, y) is not False):
            return [x, y, label]

        return False

    def spiral(x, y, args):
        half_num = args[5] * 0.5

        def genSpiral(deltaT):
            r = (args[3] % half_num) / half_num * 5
            t = 1.75 * (args[3] % half_num) / half_num * 2 * math.pi + deltaT
            x = r * math.sin(t) + random.uniform(-1, 1) * args[2]
            y = r * math.cos(t) + random.uniform(-1, 1) * args[2]
            return [x, y]

        if(args[3] > half_num):
            label = -1
            pos = genSpiral(math.pi)
        else:
            label = 1
            pos = genSpiral(0)
        pos.append(label)

        return pos

    def plane_regression(x, y, args):
        c = numpy.interp(x + y, [-10, 10], [-1, 1])
        x += random.uniform(-6, 6) * args[2]
        y += random.uniform(-6, 6) * args[2]
        return [x, y, c]

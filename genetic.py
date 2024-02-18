import pickle
import random
import time

import numpy as np
import pygame.draw

from game import Bird, SCREEN_SIZE, Game


# np.random.seed(42)
# random.seed(42)


def mul(*args):
    rst = 1
    for i in args:
        rst *= i
    return rst


class Model:
    MUTATION_PROBABILITY = 0.03
    RANDOM_FACTOR = 12
    INPUT_LENGTH = 4
    HIDDEN_LAYER_UNITS = 6

    def __init__(self, layer1=None, layer2=None, threshold=None):  # input: 横向到空隙中心的有向距离，纵向到中心的有向距离，到顶部的距离，屏幕大小
        # !!! 对传进来的数据一定要进行拷贝操作！！！
        self.layer1 = ((np.random.random((self.HIDDEN_LAYER_UNITS, self.INPUT_LENGTH)).astype('float16') - 0.5)
                       * self.RANDOM_FACTOR) if layer1 is None \
            else np.array(layer1).astype('float16')  # type:np.ndarray
        self.layer2 = ((np.random.random((1, self.HIDDEN_LAYER_UNITS)).astype('float16') - 0.5)
                       * self.RANDOM_FACTOR) if layer2 is None \
            else np.array(layer2).astype('float16')  # type:np.ndarray
        self.threshold = random.random() if threshold is None else threshold

    def calc(self, inputData):
        inputData = np.array(inputData, dtype='float16').reshape((self.INPUT_LENGTH, 1)) / SCREEN_SIZE
        return np.matmul(self.layer2, np.matmul(self.layer1, inputData)).flatten()[0]

    def cross(self, model):
        """杂交1"""
        len1 = mul(*self.layer1.shape)
        cutPoint1 = random.randint(0, len1)
        newLayer1 = np.zeros((len1,)).astype('float16')
        newLayer1[:cutPoint1] = self.layer1.flatten()[:cutPoint1]
        newLayer1[cutPoint1:] = model.layer1.flatten()[cutPoint1:]
        newLayer1 = newLayer1.reshape(self.layer1.shape)

        len2 = mul(*self.layer2.shape)
        cutPoint2 = random.randint(0, len2)
        newLayer2 = np.zeros((len2,)).astype('float16')
        newLayer2[:cutPoint2] = self.layer2.flatten()[:cutPoint2]
        newLayer2[cutPoint2:] = model.layer2.flatten()[cutPoint2:]
        newLayer2 = newLayer2.reshape(self.layer2.shape)

        threshold = random.choice((self.threshold, model.threshold))
        return Model(newLayer1, newLayer2, threshold).mutation()

    def mutation(self):
        """突变"""
        for i in range(mul(*self.layer1.shape)):
            if random.random() < self.MUTATION_PROBABILITY:
                r = i // self.layer1.shape[1]
                c = i % self.layer1.shape[1]
                self.layer1[r, c] = (random.random() - 0.5) * self.RANDOM_FACTOR

        for i in range(mul(*self.layer2.shape)):
            if random.random() < self.MUTATION_PROBABILITY:
                r = i // self.layer2.shape[1]
                c = i % self.layer2.shape[1]
                self.layer2[r, c] = (random.random() - 0.5) * self.RANDOM_FACTOR
        if random.random() < self.MUTATION_PROBABILITY:
            self.threshold = random.random()
        return self

    def copy(self):
        return Model(self.layer1, self.layer2, self.threshold)


seqCur = 0


class ModelBird(Bird):
    def __init__(self, type_="random", group=None):
        global seqCur
        super().__init__(group, seqCur)
        seqCur += 1
        self.type = type_
        self.model = Model()

    def update(self, deltaTime, userJump: bool, wallGroup) -> None:
        super().update(deltaTime, userJump, wallGroup)
        closestWall = None
        distanceHorizontal = SCREEN_SIZE + 1  # 比较的初始值
        x, y = self.rect.center
        for wall in wallGroup.objs:
            gapX, gapY = wall.getGapCenter()
            if closestWall is None and 0 < gapX - x:
                closestWall = wall
                distanceHorizontal = gapX - x
            elif 0 < gapX - x < distanceHorizontal:
                closestWall = wall
                distanceHorizontal = gapX - x
        gapX, gapY = closestWall.getGapCenter()
        if self.model.calc((gapX - x, gapY - y, self.rect.top, SCREEN_SIZE)) > self.model.threshold:
            self.jump()

    def copy(self, mutation=False):
        """创建备份，注意：seq不会备份, mutation:是否产生突变"""
        bird = ModelBird("mutation" if mutation else "copy")
        bird.model = self.model.copy()
        if mutation:
            bird.model.mutation()
        return bird

    def getFitness(self):
        return self.livingTime

    def __repr__(self):
        return f"{{ModelBird{self.seq:>10d}:{self.type:>10s}}}"


class Population:
    def __init__(self, size, kingSize, kingMultiTimes, randomSize):
        self.birds = []
        self.size = size  # 种群个体数量
        self.kingSize = kingSize
        self.kingMultipliesTimes = kingMultiTimes  # 迭代下一种群时kingBirds复制产生的倍数，剩下的是杂交bird和随机bird
        self.randomSize = randomSize  # 每一代中完全随机产生的bird的数量
        assert size >= self.kingSize * self.kingMultipliesTimes + self.randomSize, "不合理的参数导致种群个体数量不断增长"

    def initPopulation(self):
        """初代种群"""
        for i in range(self.size):
            self.birds.append(ModelBird("random"))

    def nextPopulation(self):
        """繁衍下一代"""
        ratedBirds = sorted(self.birds, key=lambda b: b.getFitness(), reverse=True)
        kingBirds = ratedBirds[:self.kingSize]
        copiedKingBirds = []
        for kb in kingBirds:
            kb.type = "king"
            for i in range(self.kingMultipliesTimes - 1):
                copiedKingBirds.append(kb.copy(mutation=True))
        randomBirds = [ModelBird("random") for i in range(self.randomSize)]
        birds = self.cross(kingBirds, self.size - self.kingSize * self.kingMultipliesTimes - self.randomSize)
        self.birds.clear()
        self.birds = kingBirds + copiedKingBirds + randomBirds + birds

    @classmethod
    def cross(cls, birdsToCross, wanted: int):
        """让指定的鸟相互交配（不包括自交）, 产生wanted个新鸟"""
        newBirds = []
        for i in range(wanted):
            chosen1, chosen2 = random.choices(birdsToCross, weights=[b.getFitness() for b in birdsToCross], k=2)
            newBird = ModelBird("cross")
            newBirds.append(newBird)
            newBird.model = chosen1.model.cross(chosen2.model)
        return newBirds


class Statistic:
    def __init__(self, size: int):
        self.arr = []
        self.size = size

    def push(self, val):
        self.arr.append(val)
        while len(self.arr) > self.size:
            self.arr.pop(0)

    def mean(self):
        return sum(self.arr) / len(self.arr)


def main():
    population = Population(1000, 30, 20, 50)
    population.initPopulation()
    game = Game(16, 100000)

    statis = Statistic(10)
    while pygame.get_init():
        # random.seed(43)
        game.main(population.birds)
        # random.seed(time.time())
        minSeqBird = min(population.birds, key=lambda b: b.seq)
        maxFitBird = max(population.birds, key=lambda b: b.getFitness())
        statis.push(game.wallCount)
        print(
            f"WallCnt:{game.wallCount:5d}, Mean:{statis.mean():9.2f}, MinSeq:{minSeqBird}, MaxFit:{maxFitBird}->{maxFitBird.getFitness():>7d}, ThisTime:{Game.getTime():>7d}")

        population.nextPopulation()

        with open("TrainedBirds.b", 'wb') as w:
            pickle.dump(population.birds[:population.kingSize], w)


if __name__ == '__main__':
    main()  # 开始训练

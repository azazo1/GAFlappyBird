import random

import numpy as np
import pygame.draw

from game import Bird, SCREEN_SIZE, Game

np.random.seed(42)
random.seed(42)


def mul(*args):
    rst = 1
    for i in args:
        rst *= i
    return rst


class Model:
    MUTATION_PROBABILITY = 0.006
    RANDOM_FACTOR = 4

    def __init__(self, layer1=None, layer2=None):
        self.layer1 = ((np.random.random((6, 2)).astype('float16') - 0.5)
                       * self.RANDOM_FACTOR) if layer1 is None else layer1  # type:np.ndarray
        self.layer2 = ((np.random.random((1, 6)).astype('float16') - 0.5)
                       * self.RANDOM_FACTOR) if layer2 is None else layer2  # type:np.ndarray
        self.fitness = -1

    def calc(self, inputData):
        inputData = np.array(inputData, dtype='float16').reshape((2, 1)) / SCREEN_SIZE
        return np.matmul(self.layer2, np.matmul(self.layer1, inputData)).flatten()[0]

    def cross(self, model):
        """杂交"""
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
        return Model(newLayer1, newLayer2).mutation()

    def mutation(self):
        """突变"""
        for i in range(mul(*self.layer1.shape)):
            if random.random() < self.MUTATION_PROBABILITY:
                r = i // self.layer1.shape[1]
                c = i % self.layer1.shape[1]
                self.layer1[r, c] = random.random() * self.RANDOM_FACTOR

        for i in range(mul(*self.layer2.shape)):
            if random.random() < self.MUTATION_PROBABILITY:
                r = i // self.layer2.shape[1]
                c = i % self.layer2.shape[1]
                self.layer2[r, c] = random.random() * self.RANDOM_FACTOR
        return self


class ModelBird(Bird):
    def __init__(self, group=None):
        super().__init__(group)
        self.model = Model()
        pygame.draw.rect(self.image, (0, 255, 0), self.image.get_rect(), width=3)

    def update(self, deltaTime, userJump: bool, wallGroup) -> None:
        super().update(deltaTime, userJump, wallGroup)
        if not self.alive():
            self.model.fitness = self.livingTime
        for wall in wallGroup.objs:
            gapX, gapY = wall.getGapCenter()
            x, y = self.rect.center
            if self.model.calc((x - gapX, y - gapY)) > 0.5:
                self.jump()


class Population:
    def __init__(self, size=100):
        self.birds = []
        self.size = size

    def initPopulation(self):
        """初代种群"""
        for i in range(self.size):
            self.birds.append(ModelBird(None))

    def nextPopulation(self):
        """繁衍下一代"""
        ratedBirds = sorted(self.birds, key=lambda b: b.livingTime, reverse=True)
        kingBirds = ratedBirds[:4]
        birds = self.cross(kingBirds, self.size - 4)
        self.birds = kingBirds + birds

    @classmethod
    def cross(cls, birdsToCross, wanted: int):
        """让指定的鸟相互交配（不包括自交）, 产生wanted个新鸟"""
        newBirds = []
        for i in range(wanted):
            chosen1, chosen2 = random.choices(birdsToCross, k=2)
            newBird = ModelBird(None)
            newBirds.append(newBird)
            newBird.model = chosen1.model.cross(chosen2.model)
        return newBirds


def main():
    population = Population(1000)
    population.initPopulation()
    game = Game(16, 10000)
    while True:
        game.main(population.birds)
        print(game.wallCount)
        population.nextPopulation()


if __name__ == '__main__':
    main()  # 开始训练

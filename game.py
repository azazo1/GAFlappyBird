import math
from random import randint

import pygame
from pygame import Surface

SCREEN_SIZE = 500
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("Consolas", 25)


class Group:
    def __init__(self):
        self.objs = []

    def addInternally(self, obj):
        self.objs.append(obj)

    def update(self, deltaTime: int, *args):
        for o in self.objs:
            o.update(deltaTime, *args)

    def draw(self, screen: Surface):
        for o in self.objs:
            o.draw(screen)

    def removeInternally(self, obj):
        self.objs.remove(obj)


# noinspection PyAttributeOutsideInit
class Bird:
    gravity = 1600  # 重力加速度 px / s / s
    # gravity = 0
    image = Surface((50, 50))
    image.fill("red")
    pygame.draw.rect(image, (0, 255, 0), image.get_rect(top=0, left=0), width=3)

    def __init__(self, group=None, seq=-1):
        self.seq = seq
        self.addGroup(group)
        self.rect = self.image.get_rect(top=0, left=100)
        self.v_Y = 0  # 纵向速度

        self.livingTime = -1

    def addGroup(self, group):
        if group:
            self.group = group
            self.group.addInternally(self)

    def removeSelfFromGroup(self):
        self.group.removeInternally(self)
        self.group = None

    def update(self, deltaTime, userJump: bool, wallGroup) -> None:
        deltaTime = deltaTime / 1000
        self.v_Y += self.gravity * deltaTime
        if userJump:
            self.jump()
        self.rect = self.rect.move(0, self.v_Y * deltaTime)
        if self.rect.top > SCREEN_SIZE or self.rect.bottom < 0:
            self.kill()
        for wall in wallGroup.objs:
            for rect in wall.rectTop, wall.rectBottom:
                if self.rect.colliderect(rect):
                    self.kill()
        if self.alive():
            # self.livingTime = Game.getTime()
            self.livingTime = max(self.livingTime, Game.getTime())  # 保留该鸟曾经最高的分数

    def alive(self):
        return bool(self.group)

    def kill(self):
        self.removeSelfFromGroup()

    def draw(self, screen: Surface):
        screen.blit(self.image, self.rect)
        text = font.render(f"{self.seq}", False, (255, 255, 0))
        rect = text.get_rect()
        rect.center = self.rect.center
        screen.blit(text, rect)

    def jump(self):
        self.v_Y = -600


# noinspection PyAttributeOutsideInit
class Wall:
    def __init__(self, group, gap: int, width=70, gapSize=140, velo=250):
        """
        :param group: Group
        :param gap: 空隙中心点y值
        :param width: 墙宽度
        :param gapSize: 空袭宽度
        :param velo: 横向向左的速度
        """
        self.addGroup(group)
        self.gap = gap
        self.velo = velo
        self.imageTop = Surface((width, gap - gapSize // 2))
        self.imageTop.fill("blue")
        self.imageBottom = Surface((width, SCREEN_SIZE - gap - gapSize // 2))
        self.imageBottom.fill("blue")
        self.rectTop = self.imageTop.get_rect(top=0, left=SCREEN_SIZE)
        self.rectBottom = self.imageBottom.get_rect(top=gap + gapSize // 2, left=SCREEN_SIZE)

    def addGroup(self, group):
        self.group = group
        self.group.addInternally(self)

    def getGapCenter(self):
        return self.rectTop.centerx, self.gap

    def removeSelfFromGroup(self):
        self.group.removeInternally(self)
        self.group = None

    def update(self, deltaTime):
        deltaTime = deltaTime / 1000
        self.rectTop.left = self.rectTop.left - deltaTime * self.velo
        self.rectBottom.left = self.rectBottom.left - deltaTime * self.velo
        if self.rectTop.right < 0:
            self.kill()

    def kill(self):
        self.removeSelfFromGroup()

    def draw(self, screen: Surface):
        screen.blit(self.imageTop, self.rectTop)
        screen.blit(self.imageBottom, self.rectBottom)


class Game:
    time = 0

    def __init__(self, fixedDeltaTime=None, fps=60):
        """
        :param fixedDeltaTime: 是否固定提供一个假的deltaTime以便游戏加速，加速的游戏可能精度下降；None为不固定，使用真实值；为整数则为要提供的固定值
        :param fps: 设置游戏帧率
        """
        self.running = True
        self.fps = fps
        self.fixedDeltaTime = fixedDeltaTime
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.wallGroup = Group()
        self.birdGroup = Group()
        self.clock = pygame.time.Clock()

        self.wallIntervalTime = 1900
        self.distanceToLastPlacement = self.wallIntervalTime + 1
        self.wallCount = 0

    @classmethod
    def getTime(cls):
        return cls.time

    def placeWall(self, deltaTime):
        self.distanceToLastPlacement += deltaTime
        if self.distanceToLastPlacement > self.wallIntervalTime:
            Wall(self.wallGroup, randint(70, SCREEN_SIZE - 70))
            self.wallCount += 1
            self.distanceToLastPlacement = 0

    def main(self, birds):
        Game.time = 0
        self.wallCount = 0
        self.distanceToLastPlacement = self.wallIntervalTime + 1
        self.birdGroup.objs.clear()
        self.wallGroup.objs.clear()

        for bird in birds:
            bird.addGroup(self.birdGroup)
            bird.rect.top = SCREEN_SIZE // 2
        while self.running:
            jump = False

            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    self.close()
                    return
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        jump = True

            tickValue = self.clock.tick(self.fps)

            if self.fixedDeltaTime is None:
                deltaTime = tickValue
            else:
                deltaTime = self.fixedDeltaTime

            Game.time += deltaTime

            self.placeWall(deltaTime)

            self.birdGroup.update(deltaTime, jump, self.wallGroup)
            self.birdGroup.draw(self.screen)

            self.wallGroup.update(deltaTime)
            self.wallGroup.draw(self.screen)

            text = font.render(f"PassedWalls:{self.wallCount}, LeftBirds:{len(self.birdGroup.objs)}", False,
                               (255, 255, 255))
            self.screen.blit(text, text.get_rect(top=10, left=10))

            if len(self.birdGroup.objs) == 0:
                return

            pygame.display.update()
            self.screen.fill((0, 0, 0))

    def close(self):
        self.running = False
        pygame.quit()


if __name__ == '__main__':
    Game().main([Bird()])  # 你自己玩

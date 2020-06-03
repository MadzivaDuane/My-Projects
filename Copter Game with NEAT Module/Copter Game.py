import random 
import time
import os
import neat 
from dask import visualize
import sys
import pygame
pygame.font.init()

#set screen parameters 
window_width = 550
window_height = 800
floor = 630

WIN = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Copter Game")
gen = 0

#create scoreboard
stat_font = pygame.font.SysFont("comicsans", 50)
end_font = pygame.font.SysFont("comicsans", 70)
draw_lines = False

#load images
path = "/Users/duanemadziva/Documents/_ Print (Hello World)/Learning Python/PythonVS/Data Science Projects/copter_images/"
copter_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path, "copter_final.png")))
base_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path, "base.png")))
obstacle_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path, "obstacle.png")))
sky_image = pygame.transform.scale2x(pygame.image.load(os.path.join(path, "sky.png")))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join(path, "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join(path, "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join(path, "bird3.png")))]

#define the helicopter classes
class Helicopter:
    image = copter_image
    max_rotation = -25
    rotation_vel = 20
    animation_time = 5

    #define __init__ class 
    def __init__(self, x, y):   #x, y are positions of the copter
        self.x = x
        self.y = y
        self.tilt = 0  #where the copter tilt starts
        self.tick_count = 0
        self.vel = 0 #copter velocity at start
        self.height = self.y #where the copter height starts
        self.img_count = 0
        self.img = self.image

    def jump(self):
        self.vel = -10.5 #the - is because the pygames coordinate system is oriented such that the top left corner is (0,0)
        self.tick_count = 0 #helps us keep track of our last jump
        self.height = self.y

    #define move class
    def move(self):
        self.tick_count += 1
        d = self.vel*self.tick_count + 1.5*self.tick_count**2  #displacement (d) from newtons laws of motion

        #now to failsafe this, and ensure the velocity does not overshoot
        if d >= 16:
             d = (d/abs(d))*16

        if d < 0:
            d -=2   #if we are moving upwards, lets move up a little more (jump)
        
        self.y = self.y + d   
        #configure upward tilt and set maximum tilt 
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
            
        else: 
            if self.tilt > -90:
                self.tilt -= self.rotation_vel

    def draw(self, win):
        self.img_count += 1
        #for fall of copter
        if self.tilt <= -80:
            self.img = self.image
            self.img_count = self.animation_time*2 #so when the copter starts going up it doesnt look akward
        #create a function to rotate the image around its center
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)  #blit just means draw

    def get_mask(self): #for collisions
        return pygame.mask.from_surface(self.img)


class Obstacle(): #y excluded because height remains relatively the same 
    obstacle_gap = 200 #gap between the 2 obstacle
    vel = 5  #velocity of obstacle

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.obstacle_top = pygame.transform.flip(obstacle_image, False, True)
        self.obstacle_bottom = obstacle_image

        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.obstacle_top.get_height()
        self.bottom = self.height + self.obstacle_gap 
    
    #set class to move the obstacle
    def move(self):
        self.x -= self.vel  #moves obstacle to the left
    
    #define the drawing of the obstacle
    def draw(self, win):
        win.blit(self.obstacle_top, (self.x, self.top))  #top obstacle 
        win.blit(self.obstacle_bottom, (self.x, self.bottom))  #bottom obstacle

    #defining a collision using pixels- when user flies into obstacle or the ground
    def collide(self, copter, win):
        copter_mask = copter.get_mask()
        top_mask = pygame.mask.from_surface(self.obstacle_top)
        bottom_mask = pygame.mask.from_surface(self.obstacle_bottom)

        top_offset = (self.x - copter.x, self.top - round(copter.y))
        bottom_offset = (self.x - copter.x, self.bottom - round(copter.y))

        b_point = copter_mask.overlap(bottom_mask, bottom_offset)
        t_point = copter_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        
        return False

#define the base of the game 
class Base():
    vel = 5
    width = base_image.get_width()
    img = base_image

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        #this essentially creates 2 base images and cycles between the two as one completes a cycle on the screen
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width
    
    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))
 
def draw_window(win, copters, obstacles, base, score, gen, obstacle_ind):  
    if gen == 0:
        gen = 1

    win.blit(sky_image, (0,0))  #draw sky in as background

    for obstacle in obstacles:
        obstacle.draw(win)  #draw obstacle 
    
    base.draw(win)

    for copter in copters:
        # draw lines from copter to obstacle
        if draw_lines:
            try:
                pygame.draw.line(win, (255,0,0), (copter.x+copter.img.get_width()/2, copter.y + copter.img.get_height()/2), (obstacles[obstacle_ind].x + obstacles[obstacle_ind].obstacle_top.get_width()/2, obstacles[obstacle_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (copter.x+copter.img.get_width()/2, copter.y + copter.img.get_height()/2), (obstacles[obstacle_ind].x + obstacles[obstacle_ind].obstacle_bottom.get_width()/2, obstacles[obstacle_ind].bottom), 5)
            except:
                pass
        # draw copter
        copter.draw(win)

    # scoreboard
    score_label = stat_font.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (window_width - score_label.get_width() - 15, 10))

    # generations
    score_label = stat_font.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = stat_font.render("Alive: " + str(len(copters)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

def main(genomes, config):

    global WIN, gen
    win = WIN
    gen += 1

    copters = []  #initialize, starting position
    neural_networks = []  #list of neral networks (nets)
    genome_list = [] #list of genomes (ge)

    for genome_id, g in genomes:
        g.fitness = 0  #set initial fitness to 0
        neural_network = neat.nn.FeedForwardNetwork.create(g, config)
        neural_networks.append(neural_network)
        copters.append(Helicopter(5, 350))  #append copter object to copters list
        genome_list.append(g)

    base = Base(700)
    obstacles = [Obstacle(680)]
    score = 0
    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                sys.exit()
                break

        obstacle_ind = 0
        if len(copters) > 0:
            if len(obstacles) > 1 and copters[0].x > obstacles[0].x + obstacles[0].obstacle_top.get_width():
                obstacle_ind = 1
        else:
            run = False
            break
        
        for x, copter in enumerate(copters):
            genome_list[x].fitness += 0.1 #give an initial fitness for copter survival 
            copter.move()  #get each copter to move
            #now to set up a nearal network for each copter
            output = neural_networks[copters.index(copter)].activate((copter.y, abs(copter.y - obstacles[obstacle_ind].height), abs(copter.y - obstacles[obstacle_ind].bottom)))
            if output[0] > 0.5:
                copter.jump()

        base.move()

        add_obstacle = False
        remove_obstacle = []
        for obstacle in obstacles:  #move more than one obstacle
            obstacle.move()

            for copter in copters:
                if obstacle.collide(copter, win):
                    genome_list[copters.index(copter)].fitness -= 1
                    neural_networks.pop(copters.index(copter))
                    genome_list.pop(copters.index(copter))
                    copters.pop(copters.index(copter))

            if obstacle.x + obstacle.obstacle_top.get_width() < 0: #if obstacle is off the screen
                remove_obstacle.append(obstacle)
            
            if not obstacle.passed and obstacle.x < copter.x:
                obstacle.passed = True
                add_obstacle = True

        if add_obstacle:
            score += 1 #if a obstacle is passed/ added, increase the score by 1
            for g in genome_list: #add fitness score for each pipe a bird passes, to encourage them to pass thru pipes rather than ramming past
                g.fitness += 5
            obstacles.append(Obstacle(window_width))

        for r in remove_obstacle:
            obstacles.remove(r)

        for copter in copters:  #check if copter hits the ground
            if copter.y + copter.img.get_height() - 10 >= floor or copter.y < -50:
                neural_networks.pop(copters.index(copter))
                genome_list.pop(copters.index(copter))
                copters.pop(copters.index(copter))

        draw_window(WIN, copters, obstacles, base, score, gen, obstacle_ind)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)  #sets population size as stipilated in configuration file

    p.add_reporter(neat.StdOutReporter(True))  #gives report on perfomance
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)   #calls the main function 50 times, evaluating the best performing copter

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)






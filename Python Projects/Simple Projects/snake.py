# python snake game
import os 
import curses
import cv2

class entity(object):

	def __init__(self,x,y):
		self.x = x
		self.y = y

class snake(object):
		
	def __init__(self):
		self.width = 40
		self.hieght = 20
		self.head = entity(19,19)
		self.food = entity(10,10)
		self.displacement = 1
		self.direction = 'right' 	
		self.tailLength = 4
		self.tail = []	
		for i in range(self.tailLength):
			temp = entity(0,0)
			self.tail.append(temp)

	def draw(self):
		for h in range(self.hieght):
			for w in range(self.width):
					if h == 0 or h == self.hieght-1:
						print('#',end='')
					else:
						if w == 0 or w == self.width-1:
							print('#',end='')
						elif w == self.head.x and h == self.head.y:
							print('0',end='')
						elif w == self.food.x and h == self.food.y:
							print('*',end='')
						else:
							check = False
							for segment in self.tail:
								if w == segment.x and h == segment.y:
									check = True
									print('o',end='')
							if check == False:
								print(' ',end='')
			print('')

	def ButtonInput(self):			
		screen.getch()

		pass

	def logic(self):
		if self.head.x >= self.width-2:
			self.head.x = 1
		if self.head.x <= 0:
			self.head.x = self.width-2
		if self.head.y >= self.hieght-1:
			self.head.y = 1
		if self.head.y <= 0:
			self.head.y = self.hieght-1
		pass

	def update(self):
		# update tail
		for i in range(self.tailLength-1,0,-1):
			self.tail[i].x = self.tail[i-1].x
			self.tail[i].y = self.tail[i-1].y
		
		self.tail[0].x = self.head.x
		self.tail[0].y = self.head.y

		if self.direction == 'right':
			self.head.x += self.displacement
		elif self.direction == 'left':
			self.head.x -= self.displacement
		elif self.direction == 'up':
			self.head.y -= self.displacement
		else:
			self.head.y += self.displacement

def delay(time):
		for t in range(1000*time): 
			pass
			
		


screen = curses.initscr()
sn = snake()
#sn.initialize(sn)	

'''for i in range(10,-1,-1):
	print(i)'''

while True:
	sn.draw()
	sn.ButtonInput()
	sn.logic()
	sn.update()
	delay(1000)
	os.system('clear')









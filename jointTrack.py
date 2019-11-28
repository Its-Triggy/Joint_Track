'''Tracks Joints'''
import numpy as np
import cv2
'''------------------------------------------CLASS---------------------------------------'''
class Joint:
	#BGR
	BLUE = (255, 0, 0)
	GREEN = (0, 255, 0)
	RED = (0, 0, 255)
	YELLOW = (0, 255, 255)
	JOINT_COLOR = YELLOW
	BONECOLOR = GREEN
	DRAW_COLOR = RED

	#Window/image size
	Ncol = 375
	Nrow = 200
	#Defines how often points are sampled (1 in every __)
	sample = 10
	#defines how large an object must be to be recognized
	minContourSize = 75
	#Defines how poorly a circle can be drawn, and still be recognized as a circle
	circleStrayTollerance = 50
	circleClosedTollerance = 30
	circleMinRadius = 30
	#Line tolerances
	lineStrayTollerance = 30
	lineSpacingTollerance = 30
	lineMinVelocity = 5
	#Defines how still a joint must be, and still be recognized as still
	stillTollerance = 10
	
	#Import cascade for face recognition 
	smile_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/smile.xml')
	fist_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/fist.xml')
	face_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml')
	face_glasses_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
	eye_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/haarcascade_eye.xml')
	upper_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/haarcascade_upperbody.xml')
	lower_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/haarcascade_lowerbody.xml')
	full_cascade = cv2.CascadeClassifier('/Users/Tristan/Desktop/opencv-3.4.1/data/haarcascades/haarcascade_fullbody.xml')
	
	#Stores all the joints that have been created
	totalJointsList = []
	
	def __init__(self, x=0, y=0, colorName='no colorName', type='',\
	lowerRange = np.array([0,0,0]), upperRange = np.array([255, 255, 255]),\
	track=1, drawTrack = 0, draw=1):
		#Coordinates of the joint
		self.x = x
		self.y = y
		#I dont think this is ever used...
		self.rad = 10
		#Name the color whatever you want
		self.colorName = colorName
		#describes if the joint searches for a color, a face, or theoretically anything else
		self.type = type
		#Describes the range of BGR colors that this joint searches for
		self.upperRange = upperRange 
		self.lowerRange = lowerRange 
		#Is this joint tracked?
		self.track = track
		#Should this joint be drawn?
		self.draw = draw
		#Should the joint tracking be drawn?
		self.drawTrack = drawTrack
		self.state = ['', 50]
		#List of stored previous positions
		self.jointTrail = []
		#defines how many previous positions are recorded
		self.trailLength = 10
		
	#Stores the last "joint.trailLength" points of joint into "jointTrail"
	def storeTrack(self):
		if len(self.jointTrail) < self.trailLength:
			if self.x != 0 or self.y !=0:
				self.jointTrail.append([self.x, self.y])
			elif len(self.jointTrail) > 0:
				self.jointTrail.remove(self.jointTrail[0])
		else:
			self.jointTrail.remove(self.jointTrail[0])
			if self.x != 0 or self.y !=0:
				self.jointTrail.append( [self.x, self.y])
	#Recognizes a still joint
	def still(self):
		jointTrail = self.jointTrail
		still = False
		if len(jointTrail) == self.trailLength:
			__, center = Joint.averagePoint(jointTrail)
			still = True
		
			for point in jointTrail:
				vector = [point[0]-center[0], point[1]-center[1]]
				magnitude = Joint.mag(vector)
				if magnitude > Joint.stillTollerance:
					still = False
		
		if still == False:
			return False
		elif still == True:
			return True
	#Recognizes circles				
	def circle(self):
		jointTrail=self.jointTrail
		radii = []
		drew_circle = False
		#Checks that there exists a full trail, and that the joint isn't simple still
		if len(jointTrail) == self.trailLength and self.still() == False:
			#This state will change as soon as a condition is not met (a point out of line, for example)
			drew_circle = True
			__, center = Joint.averagePoint(jointTrail)
			#Calculates average radius
			for point in jointTrail:
				vector = [point[0]-center[0], point[1]-center[1]]
				radii.append(Joint.mag(vector))
			ave_radius = np.mean(radii)
			#Checks that every point is the same distance away from the center 
			for point in jointTrail:
				vector = [point[0]-center[0], point[1]-center[1]]
				magnitude = Joint.mag(vector)
				if abs(magnitude-ave_radius) > Joint.circleStrayTollerance:
					drew_circle = False
				if ave_radius < Joint.circleMinRadius:
					drew_circle = False
				
			#Check to see if circle is more or less closed:
			if Joint.mag([jointTrail[0][0]-jointTrail[self.trailLength-1][0],jointTrail[0][1]-jointTrail[self.trailLength-1][1]])\
			 > Joint.circleClosedTollerance:
				drew_circle = False

			if int(ave_radius - Joint.circleStrayTollerance) > 0:
				'''
				if drew_circle == True:
					cv2.circle(Joint.img, (center[0],center[1]), int(ave_radius - Joint.circleStrayTollerance), Joint.GREEN, 2)
					cv2.circle(Joint.img, (center[0],center[1]), int(ave_radius + Joint.circleStrayTollerance), Joint.GREEN, 2)
				elif drew_circle == False:
					cv2.circle(Joint.img, (center[0],center[1]), int(ave_radius - Joint.circleStrayTollerance), Joint.RED, 2)
					cv2.circle(Joint.img, (center[0],center[1]), int(ave_radius + Joint.circleStrayTollerance), Joint.RED, 2)
				'''
		if drew_circle == False:
			return False, 0, 0
		elif drew_circle == True:
			return True, center, ave_radius
	#recognizes horizontal lines	
	def hLine(self):
		jointTrail=self.jointTrail
		vel = Joint.deriv(jointTrail)
		length = len(jointTrail)
	
		drew_hLine = False
		#Checks that there exists a full trail, and that the joint isn't simple still
		if len(jointTrail) == self.trailLength and self.still() == False:
			#This state will change as soon as a condition is not met (a point out of line, for example)
			drew_hLine = True
			for i in range(0,length-1):	
				#Checks that every point has the same vertical as point in front of it
				if abs(jointTrail[i][1]-jointTrail[i+1][1]) > Joint.lineStrayTollerance:
					drew_hLine = False
				#Ensures the joint isn't moving too slow
				if vel[i][0] < Joint.lineMinVelocity:
					drew_hLine = False
				#Check to see if line is more or less evenly spaced:
				if i<length-2:
					if abs(vel[i][0]-vel[i+1][0]) > Joint.lineSpacingTollerance:
						drew_hLine = False
		if drew_hLine == False:
			return False, 0, 0
		elif drew_hLine == True:
			return True, jointTrail[0], jointTrail[len(jointTrail)-1]
	#recognizes vertical lines	
	def vLine(self):
		jointTrail=self.jointTrail
		vel = Joint.deriv(jointTrail)
		length = len(jointTrail)
	
		drew_vLine = False
		#Checks that there exists a full trail, and that the joint isn't simple still
		if len(jointTrail) == self.trailLength and self.still() == False:
			#This state will change as soon as a condition is not met (a point out of line, for example)
			drew_vLine = True
			for i in range(0,length-1):	
				#Checks that every point has the same horizontal as point in front of it
				if abs(jointTrail[i][0]-jointTrail[i+1][0]) > Joint.lineStrayTollerance:
					drew_vLine = False
				#Ensures the joint isn't moving too slow
				if vel[i][1] < Joint.lineMinVelocity:
					drew_vLine = False
				#Check to see if line is more or less evenly spaced:
				if i<length-2:
					if abs(vel[i][1]-vel[i+1][1]) > Joint.lineSpacingTollerance:
						drew_vLine = False
		if drew_vLine == False:
			return False, 0, 0
		elif drew_vLine == True:
			return True, jointTrail[0], jointTrail[len(jointTrail)-1]
	#gathers all points of a certain color range into a list via mask
	def gatherPoints(self, img):
		mask = cv2.inRange(img, self.lowerRange, self.upperRange)
		
		sample = Joint.sample
		#collects all points containing object into a list
		white = []
		for row in range(0, Joint.Nrow, sample):
			for col in range(0, Joint.Ncol, sample):
				if mask[row, col] == 255:
					white.append([col, row])

		return mask, white	
	#Draws bones between joints
	def connectTo(self, joint2):
		if (self.x != 0 or self.y != 0) and (joint2.x != 0 or joint2.y != 0):
			cv2.line(Joint.img,(self.x,self.y),(joint2.x,joint2.y), Joint.BONECOLOR, 2)
	#Draws a trail behind a joint			
	def drawTracking(self, img):
		if len(self.jointTrail) != 0:
			point1 = self.jointTrail[0]
		if len(self.jointTrail) == 1:
			cv2.line(img,(point1[0],point1[1]),(point1[0]+2,point1[1]+2),Joint.DRAW_COLOR,2)
		elif len(self.jointTrail) > 1:
			for i in range(0, len(self.jointTrail)-1):
				cv2.line(img,(self.jointTrail[i][0],self.jointTrail[i][1]),\
				(self.jointTrail[i+1][0],self.jointTrail[i+1][1]),Joint.DRAW_COLOR,2)
	#Draws a circle at joint position
	def drawJoint(self, img):
		#Makes sure position is not default, then draws according to the state of the joint
		if self.x != 0 or self.y != 0:	
			if self.state[0] == "heal":
				cv2.circle(img, (self.x,self.y), 150, Joint.YELLOW, 80)
			if self.state[0] == "shield":
				cv2.circle(img, (self.x,self.y), 200, Joint.RED, 20)
			else:
				cv2.circle(img, (self.x,self.y), 10, Joint.JOINT_COLOR, 2)
	#Uses contouring to calculate joint position
	def findJoint_ContourMethod(self):
		mask = self.mask
		#https://www.bluetin.io/opencv/object-detection-tracking-opencv-python/
		im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
		contour_sizes = []
		#keeps track of the sizes of the contours
		for contour in contours:
			contour_sizes.append([cv2.contourArea(contour), contour])
		#Checks that the list of contours is not empty
		if contour_sizes != []:
			#Finds the biggest contour
			biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
			#Checks largest contour is large enough
			if len(biggest_contour) > Joint.minContourSize:
				#cv2.drawContours(Joint.img, biggest_contour, -1, (0,255,0), 3)
				xf,yf,w,h = cv2.boundingRect(biggest_contour)
				return Joint(x=xf+int(w/2), y=yf+int(h/2))
			else:
				#print "no SUFFICIENTLY LARGE " + str(self.colorName) + " object"
				return Joint()
		else:
			#print "no " + str(self.colorName) + " object"
			return Joint()
	#Averages all white mask points to calculate joint position
	def findJoint_AverageMethod(self):
		maskList = self.maskList
		count, ave = Joint.averagePoint(maskList)
		joint = Joint(x=ave[0],y=ave[1], lowerRange=self.lowerRange, upperRange=self.upperRange)
		return joint	
	#Finds head joint
	def findJoint_HaarMethod(self, gray):
		type = self.type
		
		if type == 'head':
			haar = Joint.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
		elif type == 'fist':
			haar = Joint.fist_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
		elif type == 'upper':
			haar = Joint.upper_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
		elif type == 'smile':
			haar = Joint.smile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

		else:
			print "joint type not recognized"
			return Joint()
		
		#if a face is recognized, a joint with its coordinates is created
		joint_found = False
		#Scans for non-glasses faces
		for (xf,yf,w,h) in haar:	
			joint_found = True
			#if type == 'fist':
			#	self.state = ['shield', 0]
			return Joint(x = xf+int(w/2), y = yf+int(h/2))
	
		#if nothing is detected, tries a different cascade
		
		if joint_found == False:
			if type == 'head':
				haar = Joint.face_glasses_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

			for (xf,yf,w,h) in haar:	
				joint_found = True
				return Joint(x = xf+int(w/2), y = yf+int(h/2))

		#If no face is detected, sets default joint (0,0)
		if joint_found == False:
			return Joint()
			
	#Returns average x and average y of a list of points
	@staticmethod
	def averagePoint(list):
		totalx = 0
		totaly = 0
		count = 0
	
		for item in list:
			totalx += item[0]
			totaly += item[1]
			count += 1
		if count > 0:
			return count, [int(totalx/count), int(totaly/count)]
		else:
			#print "div 0 avoided"
			return count, [0,0]
	#Takes derivative of a list of points (pos --> vel, vel --> acc)
	@staticmethod
	def deriv(list):
		vel = []
		#create a placeholder (zero) list
		for i in range (0, len(list)):
			vel.append("zero")
	
		#if there are 2 or more points in jointTrail, vel takes the difference between subsequent points
		if len(list) > 1:
			for i in range (0, len(list)-1):
				vel[i] = [list[i+1][0]-list[i][0], list[i+1][1]-list[i][1]]
	
		#removes leftover placeholders	
		for item in vel:
			if item == "zero":
				vel.remove("zero")
			
		return vel
	#returns the magnitude of a vector
	@staticmethod
	def mag(vector):
		x = vector[0]
		y = vector[1]
	
		return (x**2 + y**2)**.5
		
	#Updates joint positions and joint track, and draws the joints	
	def updateJoint(self, img, hsv):
		self.state[1] += 1
		
		if self.type == 'color':
			self.mask, joint.maskList = self.gatherPoints(hsv)
			self.x = (self.findJoint_ContourMethod()).x
			self.y = (self.findJoint_ContourMethod()).y
	

			if self.track ==1:
				self.storeTrack()
			if self.draw ==1:
				self.drawJoint(img)
			if self.drawTrack ==1:
				self.drawTracking(img)	
		
		else:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			self.x = (self.findJoint_HaarMethod(gray)).x
			self.y = (self.findJoint_HaarMethod(gray)).y
		
			if self.track ==1:
				self.storeTrack()
			if self.draw ==1:
				self.drawJoint(img)
			if self.drawTrack ==1:
				self.drawTracking(img)
				
'''------------------------------------------GLOBAL VARS---------------------------------'''
#BGR
#defines upper and lower range for black
LRangeBlk = np.array([0,0,0])
URangeBlk = np.array([2,2,2])

#defines upper and lower range for blue
LRangeBlu = np.array([75,0,0])
URangeBlu = np.array([255,150,100])

#defines upper and lower range for orange
LRangeOra = np.array([0,0,150])
URangeOra = np.array([200,200,255])

#defines upper and lower range for HSV blue
LRangeHSVBlu = np.array([100,0,0])
URangeHSVBlu = np.array([120,255,100])

'''------------------------------------------INIT----------------------------------------'''
#Start video capture
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
#Initialize joint

#jointBlk = Joint(lowerRange=LRangeBlk,upperRange=URangeBlk,type='color',colorName='black')
#Joint.totalJointsList.append(jointBlk)

#jointOra = Joint(lowerRange=LRangeOra,upperRange=URangeOra,type='color',colorName='orange')
#Joint.totalJointsList.append(jointOra)

#jointBlu = Joint(lowerRange=LRangeBlu,upperRange=URangeBlu,type='color',colorName='blue', drawTrack=0)
#Joint.totalJointsList.append(jointBlu)

#jointHSVBlu = Joint(lowerRange=LRangeHSVBlu,upperRange=URangeHSVBlu,type='color',colorName='hsv_blue', drawTrack=0)
#Joint.totalJointsList.append(jointHSVBlu)

jointHead = Joint(type = 'head')
Joint.totalJointsList.append(jointHead)

#jointFist = Joint(type = 'fist')
#Joint.totalJointsList.append(jointFist)



circles = []
lines = []
'''------------------------------------------MAIN LOOP-----------------------------------'''

while (True):
#setup
	#capture frame by frame
	ret, frame = cap.read()
	ret1, frame1 = cap1.read()
	#perform operation of the frame (resize and gray)
	Joint.img = cv2.resize(frame, (Joint.Ncol, Joint.Nrow))
	Joint.img1 = cv2.resize(frame1, (Joint.Ncol, Joint.Nrow))
	Joint.hsv = cv2.cvtColor(Joint.img, cv2.COLOR_BGR2HSV)
	Joint.hsv1 = cv2.cvtColor(Joint.img1, cv2.COLOR_BGR2HSV)
					
#update joints
	for joint in Joint.totalJointsList:
		joint.updateJoint(Joint.img, Joint.hsv)
		joint.updateJoint(Joint.img1, Joint.hsv1)
#processing

	#display	
	cv2.imshow('Joint.img', Joint.img)
	cv2.imshow('Joint.img1', Joint.img1)

	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
'''------------------------------------------CLOSE-------------------------------------'''
cap.release()
cv2.destroyAllWindows()
 

































'''
	#Stores circles and lines as they are drawn
	circTF, center, rad = jointBlk.circle(Joint.img)
	if circTF == True:
		circles.append([center, rad])
	hlineTF, first, last = jointBlk.hLine()
	if hlineTF == True:
		lines.append([first, last])
	vlineTF, first, last = jointBlk.vLine()
	if vlineTF == True:
		lines.append([first, last])
	
#Drawing
	
	#draws circles and lines 
	for circle in circles:
		cv2.circle(Joint.img, (int(circle[0][0]),int(circle[0][1])), int(circle[1]), Joint.GREEN, 2)
	for line in lines:
		cv2.line(Joint.img,(line[0][0],line[0][1]),(line[1][0],line[1][1]), Joint.GREEN, 2)
		

		
	#__, chest = Joint.averagePoint([[jointHead.x, jointHead.y],[jointBlk.x, jointBlk.y]])
	#__, chest2 = Joint.averagePoint([chest,[jointHead.x, jointHead.y]])
	#jointChest = Joint(x=int(chest2[0]), y = int(chest2[1]), type = '')
	#jointChest.drawJoint(Joint.img)
	#jointChest.connectTo(Joint.img, jointBlu)
	#jointChest.connectTo(Joint.img, jointOra)
	#jointChest.connectTo(Joint.img, jointBlk)
	#jointChest.connectTo(Joint.img, jointHead)
'''
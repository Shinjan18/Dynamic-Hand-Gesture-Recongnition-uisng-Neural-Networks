import pyautogui as p

def controll(data):
	x, y = p.position()
	# print(x,y)
	n=p.onScreen(x,y)
	# print(n)
	if (n == True):
		if (data == 'Blank'):
			# p.moveTo(200,200)
			print('No task')
		elif (data == "Thumbs Up"):
			p.press('down')
		elif (data == "Thumbs Down"):
			p.press('up')
		elif (data == "Punch"):
			p.press('volumedown')
		elif (data == "High Five"):
			p.press('volumeup')
		else:
			print('No task assigned')


# controll('n')
# controll("Punch")
# controll("Thumbs Down")
# controll('Blank')
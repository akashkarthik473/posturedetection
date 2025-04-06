import pyscreenshot 
  

for i in range(450, 500):
    image = pyscreenshot.grab(bbox=(10, 30, 400, 700)) 
    image.save(f"dataset/training data/training_pic {i}.png") 
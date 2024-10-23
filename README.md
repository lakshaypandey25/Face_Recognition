This is a simple code fo rface recognition for different users (user1, user2, ...).

Change the filename path etc. so that the code runs according to your device.

Create a folder 'data'.

If the Error: cv2 has not attribute 'face', then run the following commands in the terminal (of your editor):

1) pip uninstall opencv-contrib-python
2) pip install opencv-contrib-python
3) python migrate project.py

Now run the project.py by using the command: python project.py
Then run the recognize.py by using the command: python recognixe.py

When you run project.py, the program will ask for an id (1, 2, ...).
Run it for as many ids as you want.
project.py will collect dataset for different ids and store it in a folder, which itself will be saved in the folder 'data'.
Change the code so that it detects for ids other than 1 or 2.
Run recognize.py to recognize user as 1, 2, ...


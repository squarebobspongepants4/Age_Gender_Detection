#############SETTING UP THE ENVIRONMENTS,DOWNLOADING STUFFS#############
1.Ensure you are within /REPO directory

2.Run the command
	pip install -r requirements.txt

3.Go to /REPO/instructions.txt
execute the commands one by one(ommit the ! or % in the front)
or opy and run this list of commands:

	pip install thop
	cd /REPO/Real-ESRGAN
	pip install basicsr
	pip install facexlib
	pip install gfpgan
	pip install -r requirements.txt
	python setup.py develop


#.If you are having any issue with installing torch and/or torchvisoin may be you'll like to do 
'conda install pytorch torchvision -c pytorch' seperately to install torch

#############ARRANGING THE TO-BE-TESTED DATA#############

4.Put the images in /REPO/data/images folder
[the code isn't ready yet to handle videos by itself. Please transform the videos into imageframes and populate the image folder.
This code snippet can be used]

	#video frame capturer
	def FrameCapture(video_name):
  		vidObj = cv2.VideoCapture('/REPO/data/videos/'+video_name)
  		count = 0
  		success = 1
  		while success:
      			success, image = vidObj.read()
      			cv2.imwrite("/REPO/data/images/frame%d.jpg" % count, image)
      			count += 1


#############GENERATING OUTPUT#############
4.Open the run_it.ipynb notebook and run the only cell it has.
It will show the predictions in the console and will also save the predicted information in a csv file named output.csv in the folder /REPO/output




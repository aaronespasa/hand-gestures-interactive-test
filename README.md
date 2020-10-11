
Improving the world, one Zoom call at a time. 

# Inspiration 💡
Due to CoVid-19, Remote learning is becoming the only resource of learning for millions of students. Unfortunately, many teachers are also experiencing difficulties with remote learning technologies and struggle daily to create engaging lessons. 

-👉 Current polling services are frustrating to use are require to switch batch and forth between apps. 
-👉 Some students lack a suitable wifi connection. 
-👉 Audio issues are prevalent. 

# What it does 🔨
LearnBox enables teachers to increase student engagement & knowledge retention during online lectures without any extra work. It creates class polls from their questions and student reactions and uses NLP to add subtitles. All without requiring them to do anything beyond loging to Zoom. 

# How we built it 🅰️/🅱️
We used dataset of American Sign Language from Kaggle to develop a Convolutional Neural Network in Tensorflow to recognize gestures in real time. After achieving a high accuracy (85%) using various techniques such as dropout and batch normalization we realized that running 4 CNNs simultaneously would not allow us to process video in real time. We then proceeded to apply masks and keypoint estimations in OpenCV to track the hands, developing our own algorithm to classify the gestures based on the angles between the fingers. 

# Challenges
Applying all sorts of tricks to our CNN to improve its performance from 49% to 85% and moving into more complex -and computationally expensive- DL models. Creating the right framework and pipeline to be able to feed new videos recorded in zoom to our existing model and then to our website.

# Tech Stack 📚
- Lots of trial and error with Python, including:
-   OpenCV for Hand-Keypoint detection
-   Tensorflow for the CNNs
-   Seaborn for building the plots
- Javascript
- Google Cloud Natural Language API

# What is next for us
Creating a fully functional tool that enables educators hosting large lectures of 30-100 people to better interact with their students, as well as getting relevant data insights to figure out which points are getting across, and which students are falling behind. Also, our CNN model was trained on a single skin color and we would like to make it more balanced in order to make this solution suitable for millions of students and educators worldwide. 


### Resources:
- Database: https://www.kaggle.com/gti-upm/leapgestrecog

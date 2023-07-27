# SeguiOperatoreApi

The API is a Python-based system using the Django framework and utilizes the power of the Intel Realsense depth camera. 
Thanks to infrared sensors, the camera is able to calculate the distance and return the three-dimensional coordinates (x, y, z)
of a detected person. The detection of the person takes place through the use of the YOLOv8 model. 
An interesting aspect of the API is the ability to run custom training on a single person, to obtain coordinates exclusively for that specific person. 
To do this, the OpenCV and Pytorch libraries are used. The client to use this service is: "FollowOperatorClient" written in C# in the .NET development environment

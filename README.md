# FacialRecognitionService

## Issues

You may encounter the following issue when running the application:
```
QXcbConnection: Could not connect to display unix:0
```

A solution to this was found [here](https://github.com/jessfraz/dockerfiles/issues/155) and running the following command:
```
xhost +local:docker
```

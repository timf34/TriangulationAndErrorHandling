# Triangulation Logic Changelog

### Version 1.0 - 16/1/23

- Bare bones triangulation logic 
  - Homography is implemented 
  - We exclude detections beyond the bounds of the pitch, however these are still sent as 0, 0 to the triangulation 
    logic

### Version 1.1 - 18/1/23/

- Implemented `remove_oob_detections()` which removes detections that are outside the bounds of the pitch
  - Note: for the real time application, I would likely be as well off to slice the image at these bounds to reduce the
    number of pixels that need to be processed (should be faster, if the CPU overhead of slicing the image isn't too much).

### Version 1.2 - 26/1/23

- Implemented `transition_smoothing()` which smooths the transition when chaning from homography (1 camera) to 
  triangulation (> 1 camera) and vice versa
  - This is done by only moving half the way between the old position and the new position if the frames are consecutive
  - _It doesn't really have too much of a visual effect at 60FPS._

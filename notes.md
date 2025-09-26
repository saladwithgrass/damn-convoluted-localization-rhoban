# Questions

## We apply green filter to the image and segment white lines.

## How do we get exact straight lines from the segmented image?

## How do we extract features from lines?

## How do we connect two features between themselves?

## How to localize based on features?
It seems that the answer may lien in RobotFilter.cpp.

The robot uses a particle filter for localisation.
- Vision/Localisation/Field/ contains all the code relevant to the particle filter
- Vision/Binding/LocalisationBinding contains the interface of the localisation
  module with other modules


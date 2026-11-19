# Unitree G1 Apple Toss (MuJoCo)

A MuJoCo simulation of a Unitree G1 humanoid tossing an apple from its left hand to its right hand.
Made for the Statics and Dynamics course, Assignment 4, Question 3.

## Requirements
- Python 3.9+
- mujoco and numpy

## Run
pip install mujoco numpy
python throw.py

A viewer window opens and the G1 tosses and catches the apple.
Pressing Backspace in the viewer resets the simulation.

## Files
- throw.py: controls the arms and drives the toss and catch
- scene.xml: scene with floor, apple and weld constraints (includes g1.xml)
- g1.xml and assets/: Unitree G1 robot model

## Credits
- Simulation by <your friend's name> and <your name>
- Robot model: Unitree G1 from MuJoCo Menagerie
  (https://github.com/google-deepmind/mujoco_menagerie), see LICENSE

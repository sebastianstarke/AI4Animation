PFNN Demo
=========

Running
-------

This demo should run on a windows 64-bit machine without any further editing. 
Just run `pfnn.exe` (or `pfnn_hq.exe` for a higher quality version with more 
expensive rendering). If you are on any other platform you will probably have 
to recompile (see below). You need to plug in an Xbox gamepad before you start 
the program, and the controls are given as follows.


    Keyboard Number Keys   -> Change Map
    Gamepad Left Stick     -> Move Character
    Gamepad Right Stick    -> Move Camera
    Gamepad Left Trigger   -> Strafe (Face Camera)
    Gamepad Right Trigger  -> Jog / Run
    Gamepad Left Shoulder  -> Zoom In
    Gamepad Right Shoulder -> Zoom Out
    Gamepad B              -> Toggle Crouch
    Gamepad X              -> Toggle Debug Display
    Gamepad Back           -> Toggle Extra Resonsiveness
    Gamepad Start          -> Toggle IK


On some windows machines the controls may be messed up, in which case you may 
need to play with the gamepad enums at the top of `pfnn.cpp` and recompile.


Compilation
-----------

All of the source code for the demo is contained in `pfnn.cpp`. It has the 
following dependancies which you will need to install before compiling:

    glm
    glew
    SDL2
    Eigen

With all these libraries installed you should be able to run the following to 
compile:

    make

If you get any errors you may have to adjust the Makefile a bit. Once compiled
sucessfully you can run the demo:

    ./pfnn
    
There is also a version with higher quality rendering which was used to produce
the video. The rendering code is quite inefficient in this version so it 
requires a half decent graphics card to run properly.

    ./pfnn_hq
    
There are various options at the top of the file you can play around with. The 
network weights are in the folder `network/pfnn`. These are the ones used in 
the video and trained using the process described in the paper. Various things 
including the skeleton structure are somewhat hard-coded so be careful if you 
are using your own data and own network weights and want to update this demo 
too.





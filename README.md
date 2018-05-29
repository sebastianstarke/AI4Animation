AI4Animation
======================================================

Copyright Information
------------
This code implementation is only for research or education purposes, and (especially the learned data) not freely available for commercial use or redistribution. The intellectual property and code implementation belongs to the University of Edinburgh. Licensing is possible if you want to apply this research for commercial use. For scientific use, please reference this repository together with the relevant publications below. In any case, I would ask you to contact me if you intend to seriously use, redistribute or publish anything related to this code or repository.

Description
------------
This project explores the opportunities of deep learning and artificial intelligence for character animation and control as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by Taku Komura and in collaboration with He Zhang and Jun Saito. The development is done using Unity3D / Tensorflow, and the implementations are made available during my Ph.D. progress.

SIGGRAPH 2018
------------
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2018/Teaser.png" width="100%">
Animating characters is a pain, right? Especially those four-legged monsters!
This year, we will be presenting our recent research on quadruped animation and character control at the SIGGRAPH 2018 in Vancouver.
The system can produce natural animations from real motion data using a novel neural network architecture, called Mode-Adaptive Neural Networks.
Instead of optimising a fixed group of weights, the system learns to dynamically blend a group of weights into a further neural network, based on the current
state of the character. That said, the system does not require labels for the phase or locomotion gaits, but can learn from unstructured motion capture data in an
end-to-end fashion.
<br /><br />

[![Mode-Adaptive Neural Networks for Quadruped Motion Control](https://img.youtube.com/vi/uFJvRYtjQ4c/0.jpg)](https://www.youtube.com/watch?v=uFJvRYtjQ4c)

<div align="center">
    <iframe width="620" height="315"
        src="https://www.youtube.com/embed/OhlVBpEnjig">
    </iframe>
</div>

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2018/Terrain.png" width="100%">

Paper: https://github.com/sebastianstarke/AI4Animation//blob/master/Media/SIGGRAPH_2018/Paper.pdf
<br /><br />
A demo of our system will be availble in this repository very soon.

SIGGRAPH 2017
------------
This work continues the recent work on PFNN (Phase-Functioned Neural Networks) for character control.

Video: https://www.youtube.com/watch?v=Ul0Gilv5wvY

Paper: http://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf

A demo in Unity3D using the original weights for terrain-adaptive locomotion is contained in the Assets/Demo/SIGGRAPH_2017/Original folder.
Another demo on flat ground using the Adam character is contained in the Assets/Demo/SIGGRAPH_2017/Adam folder.
In order to run them, you need to download the neural network weights from the link provided in the Link.txt file, extract them into the /NN folder, 
and store the parameters via the custom inspector button.
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2017/Adam.png" width="100%">
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2017/Original.png" width="100%">

Motion Capture
------------
In progress.
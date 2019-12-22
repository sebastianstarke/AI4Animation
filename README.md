AI4Animation
======================================================

Description
------------
This project explores the opportunities of deep learning for character animation and control as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by <a href="http://homepages.inf.ed.ac.uk/tkomura">Taku Komura</a>. Over the last couple years, this project has become a modular and stable framework for data-driven character animation, including data processing, network training and runtime control, developed in Unity3D / Tensorflow / PyTorch. This repository enables using neural networks for animating biped locomotion, quadruped locomotion, and character-scene interactions with objects and the environment. Further advances on this research will continue to be added to this project.

SIGGRAPH Asia 2019<br />
Neural State Machine for Character-Scene Interactions<br >
<sub>Sebastian Starke*, He Zhang*, Taku Komura, Jun Saito. ACM Trans. Graph. 38, 6, Article 178.</sub><br /><sub><sup>(*Joint First Authors)</sup><sub>
------------
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_Asia_2019/Teaser.jpg" width="100%">
Animating characters can be an easy or difficult task - interacting with objects is one of the latter.
In this research, we present the Neural State Machine, a data-driven deep learning framework for character-scene interactions. The difficulty in such animations is that they require complex planning of periodic as well as aperiodic movements to complete a given task. Creating them in a production-ready quality is not straightforward and often very time-consuming. Instead, our system can synthesize different movements and scene interactions from motion capture data, and allows the user to seamlessly control the character in real-time from simple control commands. Since our model directly learns from the geometry, the motions can naturally adapt to variations in the scene. We show that our system can generate a large variety of movements, icluding locomotion, sitting on chairs, carrying boxes, opening doors and avoiding obstacles, all from a single model. The model is responsive, compact and scalable, and is the first of such frameworks to handle scene interaction tasks for data-driven character animation.<br /><br />

<p align="center">
-
<a href="https://www.youtube.com/watch?v=7c6oQP1u2eQ">Video</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation//blob/master/Media/SIGGRAPH_Asia_2019/Paper.pdf">Paper</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_Asia_2019">Code & Data</a>
-
Demos (coming soon)
-
<!-- <a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Windows.zip">Windows Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Linux.zip">Linux Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Mac.zip">Mac Demo</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/Assets/Demo/SIGGRAPH_2018/ReadMe.txt">ReadMe</a>
- -->
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=7c6oQP1u2eQ">
<img width="60%" src="https://img.youtube.com/vi/7c6oQP1u2eQ/0.jpg">
</a>
</p>

SIGGRAPH 2018<br />
Mode-Adaptive Neural Networks for Quadruped Motion Control<br >
<sub>He Zhang*, Sebastian Starke*, Taku Komura, Jun Saito. ACM Trans. Graph. 37, 4, Article 145.</sub><br /><sub><sup>(*Joint First Authors)</sup><sub>
------------
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2018/Teaser.png" width="100%">
<img align="left" src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2018/Terrain.png" width="35%">
Animating characters can be a pain, especially those four-legged monsters!
This year, we will be presenting our recent research on quadruped animation and character control at the SIGGRAPH 2018 in Vancouver. The system can produce natural animations from real motion data using a novel neural network architecture, called Mode-Adaptive Neural Networks. Instead of optimising a fixed group of weights, the system learns to dynamically blend a group of weights into a further neural network, based on the current state of the character. That said, the system does not require labels for the phase or locomotion gaits, but can learn from unstructured motion capture data in an end-to-end fashion.<br /><br /><br />

<p align="center">
-
<a href="https://www.youtube.com/watch?v=uFJvRYtjQ4c">Video</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation//blob/master/Media/SIGGRAPH_2018/Paper.pdf">Paper</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018">Code</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/MotionCapture.zip">Mocap Data</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Windows.zip">Windows Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Linux.zip">Linux Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Mac.zip">Mac Demo</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018/ReadMe.txt">ReadMe</a>
-
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=uFJvRYtjQ4c">
<img width="60%" src="https://img.youtube.com/vi/uFJvRYtjQ4c/0.jpg">
</a>
</p>

SIGGRAPH 2017<br />
Phase-Functioned Neural Networks for Character Control<br >
<sub>Daniel Holden, Taku Komura, Jun Saito. ACM Trans. Graph. 36, 4, Article 42.</sub>
------------
<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2017/Adam.png" width="100%">
<img align="left" src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2017/Original.png" width="45%">
This work continues the recent work on PFNN (Phase-Functioned Neural Networks) for character control.
A demo in Unity3D using the original weights for terrain-adaptive locomotion is contained in the Assets/Demo/SIGGRAPH_2017/Original folder.
Another demo on flat ground using the Adam character is contained in the Assets/Demo/SIGGRAPH_2017/Adam folder.
In order to run them, you need to download the neural network weights from the link provided in the Link.txt file, extract them into the /NN folder, 
and store the parameters via the custom inspector button.<br /><br /><br />

<p align="center">
-
<a href="https://www.youtube.com/watch?v=Ul0Gilv5wvY">Video</a>
-
<a href="http://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf">Paper</a>
-
<a href="https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2017">Code (Unity)</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2017/Demo_Windows.zip">Windows Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2017/Demo_Linux.zip">Linux Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2017/Demo_Mac.zip">Mac Demo</a>
-
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=Ul0Gilv5wvY">
<img width="60%" src="https://img.youtube.com/vi/Ul0Gilv5wvY/0.jpg">
</a>
</p>

Processing Pipeline
------------
In progress. More information will be added soon.

<img src ="https://github.com/sebastianstarke/AI4Animation/blob/master/Media/ProcessingPipeline/Editor.png" width="100%">

Copyright Information
------------
This code implementation is only for research or education purposes, and (especially the learned data) not freely available for commercial use or redistribution. The intellectual property and code implementation belongs to the University of Edinburgh and Adobe Systems. Licensing is possible if you want to apply this research for commercial use. For scientific use, please reference this repository together with the relevant publications below. In any case, I would ask you to contact me if you intend to seriously use, redistribute or publish anything related to this repository.
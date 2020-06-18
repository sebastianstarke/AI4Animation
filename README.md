AI4Animation: Deep Learning, Character Animation, Control
============

This project explores the opportunities of deep learning for character animation and control as part of my Ph.D. research at the University of Edinburgh in the School of Informatics, supervised by <a href="http://homepages.inf.ed.ac.uk/tkomura">Taku Komura</a>. Over the last couple years, this project has become a modular and stable framework for data-driven character animation, including data processing, network training and runtime control, developed in Unity3D / Tensorflow / PyTorch. This repository enables using neural networks for animating biped locomotion, quadruped locomotion, and character-scene interactions with objects and the environment, plus sports games. Further advances on this research will continue being added to this project.

<p align="center">
<a href="https://www.youtube.com/watch?v=wNqpSk4FhSw">
<img width="60%" src="https://img.youtube.com/vi/wNqpSk4FhSw/0.jpg">
</a>
</p>

------------
**SIGGRAPH 2020**<br />
**Local Motion Phases for Learning Multi-Contact Character Movements**<br >
<sub>
<a href="https://www.linkedin.com/in/sebastian-starke-b281a6148/">Sebastian Starke</a>, 
<a href="https://www.linkedin.com/in/evan-yiwei-zhao-18584a105/">Yiwei Zhao</a>, 
<a href="https://www.linkedin.com/in/taku-komura-571b32b/">Taku Komura</a>, 
<a href="https://www.linkedin.com/in/kazizaman/">Kazi Zaman</a>.
ACM Trans. Graph. 39, 4, Article 54.
<sub>
------------
<img src ="Media/SIGGRAPH_2020/Teaser.png" width="100%">

<p align="center">
Not sure how to align complex character movements? Tired of phase labeling? Unclear how to squeeze everything into a single phase variable? Don't worry, a solution exists!
</p>
<p align="center">
<img src ="Media/SIGGRAPH_2020/Court.jpg" width="60%">
</p>

<p align="center">
Controlling characters to perform a large variety of dynamic, fast-paced and quickly changing movements is a key challenge in character animation. In this research, we present a deep 
learning framework to interactively synthesize such animations in high quality, both from unstructured motion data and without any manual labeling. We introduce the concept of local 
motion phases, and show our system being able to produce various motion skills, such as ball dribbling and professional maneuvers in basketball plays, shooting, catching, avoidance, 
multiple locomotion modes as well as different character and object interactions, all generated under a unified framework.
</p>

<p align="center">
-
<a href="https://www.youtube.com/watch?v=Rzj3k3yerDk">Video</a>
-
<a href="Media/SIGGRAPH_2020/Paper.pdf">Paper</a>
-
Code (coming soon)
-
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=Rzj3k3yerDk">
<img width="60%" src="https://img.youtube.com/vi/Rzj3k3yerDk/0.jpg">
</a>
</p>

------------
**SIGGRAPH Asia 2019**<br />
**Neural State Machine for Character-Scene Interactions**<br >
<sub>
<a href="https://www.linkedin.com/in/sebastian-starke-b281a6148/">Sebastian Starke</a><sup>+</sup>, 
<a href="https://www.linkedin.com/in/he-zhang-148467165/">He Zhang</a><sup>+</sup>, 
<a href="https://www.linkedin.com/in/taku-komura-571b32b/">Taku Komura</a>, 
<a href="https://www.linkedin.com/in/jun-saito/">Jun Saito</a>. 
ACM Trans. Graph. 38, 6, Article 178.
</sub><br /><sub><sup>(+Joint First Authors)</sup>
<sub>
------------
<img src ="Media/SIGGRAPH_Asia_2019/Teaser.jpg" width="100%">

<p align="center">
Animating characters can be an easy or difficult task - interacting with objects is one of the latter.
In this research, we present the Neural State Machine, a data-driven deep learning framework for character-scene interactions. The difficulty in such animations is that they require complex planning of periodic as well as aperiodic movements to complete a given task. Creating them in a production-ready quality is not straightforward and often very time-consuming. Instead, our system can synthesize different movements and scene interactions from motion capture data, and allows the user to seamlessly control the character in real-time from simple control commands. Since our model directly learns from the geometry, the motions can naturally adapt to variations in the scene. We show that our system can generate a large variety of movements, icluding locomotion, sitting on chairs, carrying boxes, opening doors and avoiding obstacles, all from a single model. The model is responsive, compact and scalable, and is the first of such frameworks to handle scene interaction tasks for data-driven character animation.
</p>

<p align="center">
-
<a href="https://www.youtube.com/watch?v=7c6oQP1u2eQ">Video</a>
-
<a href="Media/SIGGRAPH_Asia_2019/Paper.pdf">Paper</a>
-
<a href="AI4Animation/SIGGRAPH_Asia_2019">Code & Data</a>
-
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=7c6oQP1u2eQ">
<img width="60%" src="https://img.youtube.com/vi/7c6oQP1u2eQ/0.jpg">
</a>
</p>

------------
**SIGGRAPH 2018**<br />
**Mode-Adaptive Neural Networks for Quadruped Motion Control**<br >
<sub>
<a href="https://www.linkedin.com/in/he-zhang-148467165/">He Zhang</a><sup>+</sup>, 
<a href="https://www.linkedin.com/in/sebastian-starke-b281a6148/">Sebastian Starke</a><sup>+</sup>, 
<a href="https://www.linkedin.com/in/taku-komura-571b32b/">Taku Komura</a>, 
<a href="https://www.linkedin.com/in/jun-saito/">Jun Saito</a>. 
ACM Trans. Graph. 37, 4, Article 145.
</sub><br /><sub><sup>(+Joint First Authors)</sup>
<sub>
------------
<img src ="Media/SIGGRAPH_2018/Teaser.png" width="100%">

<p align="center">
Animating characters can be a pain, especially those four-legged monsters!
This year, we will be presenting our recent research on quadruped animation and character control at the SIGGRAPH 2018 in Vancouver. The system can produce natural animations from real motion data using a novel neural network architecture, called Mode-Adaptive Neural Networks. Instead of optimising a fixed group of weights, the system learns to dynamically blend a group of weights into a further neural network, based on the current state of the character. That said, the system does not require labels for the phase or locomotion gaits, but can learn from unstructured motion capture data in an end-to-end fashion.
</p>

<p align="center">
-
<a href="https://www.youtube.com/watch?v=uFJvRYtjQ4c">Video</a>
-
<a href="Media/SIGGRAPH_2018/Paper.pdf">Paper</a>
-
<a href="AI4Animation/SIGGRAPH_2018">Code</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/MotionCapture.zip">Mocap Data</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Windows.zip">Windows Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Linux.zip">Linux Demo</a>
-
<a href="http://www.starke-consult.de/UoE/GitHub/SIGGRAPH_2018/Demo_Mac.zip">Mac Demo</a>
-
<a href="AI4Animation/SIGGRAPH_2018/ReadMe.txt">ReadMe</a>
-
</p>

<p align="center">
<a href="https://www.youtube.com/watch?v=uFJvRYtjQ4c">
<img width="60%" src="https://img.youtube.com/vi/uFJvRYtjQ4c/0.jpg">
</a>
</p>

------------
**SIGGRAPH 2017**<br />
**Phase-Functioned Neural Networks for Character Control**<br >
<sub>
<a href="https://www.linkedin.com/in/daniel-holden-300b871b/">Daniel Holden</a>,
<a href="https://www.linkedin.com/in/taku-komura-571b32b/">Taku Komura</a>, 
<a href="https://www.linkedin.com/in/jun-saito/">Jun Saito</a>. 
ACM Trans. Graph. 36, 4, Article 42.
</sub>
------------
<img src ="Media/SIGGRAPH_2017/Adam.png" width="100%">

<p align="center">
This work continues the recent work on PFNN (Phase-Functioned Neural Networks) for character control.
A demo in Unity3D using the original weights for terrain-adaptive locomotion is contained in the Assets/Demo/SIGGRAPH_2017/Original folder.
Another demo on flat ground using the Adam character is contained in the Assets/Demo/SIGGRAPH_2017/Adam folder.
In order to run them, you need to download the neural network weights from the link provided in the Link.txt file, extract them into the /NN folder, 
and store the parameters via the custom inspector button.
</p>

<p align="center">
-
<a href="https://www.youtube.com/watch?v=Ul0Gilv5wvY">Video</a>
-
<a href="http://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf">Paper</a>
-
<a href="AI4Animation/SIGGRAPH_2017">Code (Unity)</a>
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

------------

Processing Pipeline
============
In progress. More information will be added soon.

<img src ="Media/ProcessingPipeline/Editor.png" width="100%">

Copyright Information
============
This project is only for research or education purposes, and not freely available for commercial use or redistribution. The intellectual property for different scientific contributions belongs to the University of Edinburgh, Adobe Systems and Electronic Arts. Licensing is possible if you want to use the code for commercial use. For scientific use, please reference this repository together with the relevant publications below.

The motion capture data is available only under the terms of the [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/legalcode) (CC BY-NC 4.0) license.
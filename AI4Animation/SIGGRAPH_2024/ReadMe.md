# Categorical Codebook Matching for Embodied Character Controllers
This repository contains the code for the codebook-matching technology.
Any questions, feel free to ask. For any issues you might find, please let us know and send us a message

sebastian.starke@mail.de, paulstarke.ps@gmail.com.

## Getting Started

1. Clone this repository.

2. Download the processed [MotionAssets](https://starke-consult.de/AI4Animation/SIGGRAPH_2024/MotionAssets.zip) of the cranberry motion capture dataset.

3. Extract  `MotionAssets.zip` to `Assets/Projects/CodebookMatching/MotionCapture`.

4. Download the trained [Neural Networks](https://starke-consult.de/AI4Animation/SIGGRAPH_2024/Networks.zip).

5. Extract  `Networks.zip`  to `Assets/Projects/CodebookMatching/`.

We provide three demos with different setups in `Assets/Projects/CodebookMatching/Demo_*`. 

## Standalone Mouse Input Demo
An already compiled demo is available for [Windows](https://starke-consult.de/AI4Animation/SIGGRAPH_2024/Demo_Win.zip) and [Mac](https://starke-consult.de/AI4Animation/SIGGRAPH_2024/Demo_Mac.zip) where you can try out the system using simple mouse input to control the character. Just download, unzip and start the executable in the demo folder (ReadMe.txt file included) or open them inside the Unity Editor in `Assets/Projects/CodebookMatching/Demo_Standalone`.
The menu allows you to select between a locomotion control and hybrid control demo. You will find details on how to control the character movements with mouse input inside each demo application in the info panel at the top. Both demos are expected to run at 30fps.

**Windows:** Launch the executable (Unity.exe) \
**Mac:** You might need to run `chmod +x MacDemo.app/Contents/MacOS/Unity` to launch the application.
If the app cannot be opened because Apple cannot check it for malicious software, go to Pricacy & Security settings and manually allow the app execution.

### Locomotion Control Scene (Untracked Upper-body)
This demo showcases how our codebook matching technique allows synthesizing a diverse range of realistic and responsive locomotion behaviors.

Visualize the following by pressing the corresponding buttons at the top:\
    `Draw Control` &rarr; Target position and orientation as well as the future root trajectory.\
    `Draw Sequence` &rarr; Predicted future motion sequence that is sampled from the codebook.\
    `Draw History` &rarr; One second motion history of the skeleton.\
    `Draw Codebook` &rarr; The selected categorical sample (right) and corresponding codebook vector (left) at each frame.

Adjust the parameters (KNN, Rollout) and observe how the movements change:\
    `KNN` &rarr; Sampling N motion candidates from the codebook. Higher value can help increasing diversity and stability of the generated movements\
    `Rollout` &rarr; Rolling out multiple poses from the predicted sequences may smooth the motion but degrade responsiveness.

### Hybrid Control Scene (Tracked Upper-body)
This demo shows how our system enables mixing three-point inputs with additional input modalities such as mouse input, joystick or button. Here, you can apply our hybrid control scheme to a large variety of tracker inputs by selecting one of the six reference assets on the left side. When not using mouse input, the character will try to perform the reference motion using 3PT input. By using additional mouse input you can translate the generated movements.

Similarly, you may visualize the following by pressing the corresponding buttons at the top:\
    `Draw Control` &rarr; Target position and orientation as well as the future root trajectory.\
    `Draw Body Prediction` &rarr; Upper body prediction on top of the reference character.\
    `Draw Future Prediction` &rarr; Future motion prediction on top of the reference character.

When changing the reference motions or after full progression, the character is expected to reset.
For the demos, note that the motion apperances may depend on how the user controls the position and 
orientation of the character. This means, a little practise rotating the target when moving with the 
left mouse button will likely impact the quality of those.

## VR Demo - Three Point Tracking And Hybrid Mode
In this demo you can control your virtual character with a VR device such as Quest2 or Quest3 using the three-point inputs of headset and controller and mixing them additionally with the joystick input of the controller.

### Install on Device
1. Download the <a href="https://starke-consult.de/AI4Animation/SIGGRAPH_2024/VR Demo.zip">VR Demo</a>
2. Install the .apk to your headset device for example via the `Meta Quest Developer Hub`.\
The app can then be found under `Library` &rarr; `Applications` &rarr; `Installed Prototypes` on your device.
3. You need to have both controllers and the headset active to run the app. 
After entering the app, press X on your left controller to calibrate. The calibration done is based on your head position. 
We have two modes, three-pt tracking mode and hybrid mode. With the hybrid mode, you can move the character with your left hand joystick. 

`Controls` \
X: Height Calibration. Make sure to look straight ahead.\
A: Three-Point Tracking mode (driven with headset + controllers)\
B: Hybrid mode (headset + controllers + joystick)\
Left Hand Joystick: move the hybrid character around.

### Streaming from PC
1. Open the demo scene inside the Unity Editor in `Assets/Projects/CodebookMatching/Demo_ThreePointTracking`.
2. Connect your VR device with a link-cable to your machine. For Quest and Windows you'll need the Quest desktop app to access link mode. Make sure to use the same Meta account on your VR device and machine.
3. Enable link-mode on your VR device and start `PlayMode` in the Unity Editor of your machine.
4. Calibrate your height by pressing the primary button of your left controller (Quest2 = X button). It is important to look forward in a straight line to reduce calibration errors such as when looking down or tilting your head.
5. You can now walking freely in VR. Use the joystick of your left controller to switch to hybrid mode where you can walk around in the virtual world while your real upper-body movement is embedded.

## Demo Scenes to test quality of trained models
We provide three demo scenes in `Assets/Projects/CodebookMatching/Demo_TrainingTest` where you can test the behavior of the individual models when reproducing the results:

----

`TrackingSystem_Only.unity`\
In this demo you can reconstruct the current state of our upper-body and estimate a future of the character movements by just using three point inputs.
In the scene you can test if the `TrackingSystem` component works as expected by playing back a reference motion with the `MotionEditor`.
The TrackingSystem is responsible for tracking the upper-body movements and includes two networks that you can switch out in the Inspector:
1. `Tracking Network` predicts the current upper-body pose and a root update, given the three-point signal history. (MLP)
2. `Future Network` estimates a future upper-body and root motion. (MLP)

You can check the predictions by starting the `PlayMode` and select different reference motions in the MotionEditor. In the TrackingSystem Inspector you can visualize the features in the `Drawing` section.
Expected is that the red upper-body as well as the root transformation is aligned with the played-back character movements.

----

`ThreePointTracking_ReferenceClipViaMotionEditor.unity`\
In this demo you can animate the full-body motion of the character by playing-back a reference clip in the `MotionEditor`.
You can check how well the predicted character movements (in black) match the ground truth character movements (in grey) by starting the `PlayMode` and select different reference motions in the MotionEditor.\
The `MotionController` is responsible for generating the full-body movements and includes two networks that you can switch out in the Inspector:
1. `Lower-Body Network` predicts the future sequences of lower-body movements autoregressively and using our future control signal estimates from the TrackingSystem. (CM)
2. `Tracked-Upperbody Network` predicts the current upper-body given the tracker and lower-body history. (MLP)

----

`ThreePointTracking_MoveTrackingSystemViaMouseDrag.unity`\
In this demo you can animate the full-body motion of the character by moving the `TrackingSystem` gameobject in the scene. Change position and rotation of the transform and the lower-body of the character will be generated accordingly.

## How to reproduce the results?
The complete code that was used for processing, training, and generating the movements is provided in this repository.
To reproduce the results and models complete the following steps:

1. Open `Assets/Projects/CodebookMatching/MotionCapture/Scene.unity`. <br>
2. Click on the MotionEditor gameobject in the scene hierarchy window. <br>
3. Open the AssetPipeline `Header -> AI4Animation -> Tools -> AssetPipeline` and set up as follows:<br>
    `Setup` &rarr; `ExportPipeline`<br>
    `Write Mirror` &rarr; true/checked<br>
    `Subsample Target Framerate` &rarr; true/checked<br>
    `Sequence Length` &rarr; 15<br>
    `Mode` &rarr; The option for which network you want to export training data for. More details for each network can be found in the paper or in the above section `Demo Scenes to test impact of trained models`<br>
4. Click the `Refresh` and `Process` button, which will generate the training data and save it to `SIGGRAPH_2024/PyTorch/Dataset` <br>
5. Create a new folder with the corresponding network name in `PyTorch/Datasets` and paste the dataset in this folder from `/Dataset`.
6. Navigate to `PyTorch/Models/CodebookMatching` or `PyTorch/Models/MultiLayerPerceptron` depending what Network you want to train. <br>
7. Run `Network.py` which will start the training. <br>
8. Wait for a few hours. <br> 
9. You will find the trained .onnx model in the training folder. We recommend using a 10,30,70, or 150 epoch trained model. <br>
10. Import the model into Unity and link it to the controller or tracking system. See section `Demo Scenes to test impact of trained models`.<br>
11. Hit Play.

----
When deciding to start from the raw motion capture and not use the already processed motion assets in Unity, download the [cranberry dataset](https://starke-consult.de/AI4Animation/SIGGRAPH_2024/Cranberry_Dataset.zip) and complete the  following steps:

1. Open `Assets/Projects/CodebookMatching/MotionCapture/Scene.unity`. <br>
2. Import the motion data into Unity by opening the FBX Importer `Header -> AI4Animation -> Importer -> FBX Importer`. Enter the absolute source path where the original .fbx data is saved on your hard disk, and where the motion assets should be saved inside the project (we suggest `Assets/Projects/CodebookMatching/MotionCapture/Assets`).
3. Click the `Load Source Directory` and `Process` button, which will generate import the data and save it in the destination folder. Wait until the motion data is imported.<br>
4. In the scene go to the MotionEditor component.
5. Input the path where the imported motion data assets have been saved and click `Import`.
6. Open the AssetPipeline `Header -> AI4Animation -> Tools -> AssetPipeline` and click the `Refresh` and `Process` button, which will pre-process the motion data and add multiple modules to each motion asset.
8. Wait until end of processing.
9. At this point, the raw motion capture data has been successfully processed and is at the same stage as the motion assets provided in this repository.\
You are ready to continue with the steps above to export the datasets and train the networks.

## FAQ
Q: What Unity version should I install? \
A: We suggest using Unity Version 2022.3.11f1.

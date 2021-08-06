MeshSkinnerDemo 1.0.1 (Winterdust, Sweden)

Extract the ZIP archive in the "Assets" folder inside your Unity project.
This path should exist after you've extracted: Assets/Plugins/Winterdust

Afterwards you can put this at the top of your scripts to access the MeshSkinner class:

	using Winterdust;

Getting started is simple. Here's an example line of code to try:

	new MeshSkinner(GameObject.Find("My_Model"), GameObject.Find("My_Skeleton_Container")).work().debug().finish();

What it does is find the two required GameObjects, prepare them, work, add a DebugWeights component to the model and finally apply the work (make the model skinned).
The My_Model GameObject contains one or more meshes. The My_Skeleton_Container contains the skeleton hierarchy (a bunch of empty GameObjects).

When benchmarking remember that the performance of the DLL will increase a lot when you run from a stand-alone build.
The work() method will usually finish around twice as fast outside the Unity Editor.

The demo version has the following limitations:

+ You have to put an empty text file into Assets/Resources by a specific name.
  The first time you make an instance of the MeshSkinner class it will print the required name in the console.
  This name needs to be changed every once in a while. You'll be informed by the MeshSkinner's constructor when the time comes.

Thanks for checking out MeshSkinnerDemo! Get the full version today!

https://winterdust.itch.io/meshskinner
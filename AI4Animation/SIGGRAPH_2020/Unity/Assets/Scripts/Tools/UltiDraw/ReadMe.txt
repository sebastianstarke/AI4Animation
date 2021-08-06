========================================================================================================================
------------------------------------------------------- UltiDraw -------------------------------------------------------
========================================================================================================================

This asset is an easy-to-use and light-weighted drawing library for visualising lines, primitives, gizmos or GUI elements
in edit mode and runtime. Everything you need is:
# Ultidraw.cs file
# Resources folder

The elements are supposed to be drawn in the following functions:
# OnDrawGizmos (edit mode)
# OnRenderObject (runtime)

UltiDraw assumes the following scripting pattern in order to perform computationally fast drawing calls:
# UltiDraw.Begin();
# <Drawing Call 1...>
# <Drawing Call 2...>
# <Drawing Call 3...>
# UltiDraw.End();

The following drawing features are implemented to allow visual customisation:
# Coloring
# Solid Objects
# Wire Objects
# Combined Wired (Solid+Wire) Objects
# Depth and X-Ray Rendering (Boolean Flag)
# Spatiality Shading (Parameter between 0 and 1)
# Curvature Shading (Parameter between 0 and 1)

The following drawing elements are so far implemented:
# Line (Variable Start and End Thickness)
# Circle
# Ellipse
# Arrow
# Grid
# Quad
# Cube
# Cuboid
# Sphere
# Ellipsoid
# Cylinder
# Capsule
# Cone
# Pyramid
# Bone
# Translate / Rotate / Scale Gizmos (Non-Interactable)
# Mesh
# GUI Line
# GUI Rectangle
# GUI Triangle
# GUI Circle
# GUI Function

If you require any further elements or functionalities for your projects, please let me know.
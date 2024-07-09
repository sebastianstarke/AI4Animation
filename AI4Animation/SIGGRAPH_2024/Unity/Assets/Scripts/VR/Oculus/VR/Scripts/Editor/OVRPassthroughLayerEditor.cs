/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;

using UnityEditor;
using UnityEngine;

using ColorMapEditorType = OVRPassthroughLayer.ColorMapEditorType;

[CustomPropertyDrawer(typeof(OVRPassthroughLayer.SerializedSurfaceGeometry))]
class SerializedSurfaceGeometryPropertyDrawer : PropertyDrawer
{
    public override void OnGUI(Rect rect, SerializedProperty property, GUIContent label)
    {
        // Find the SerializedProperties by name
        var meshFilterProperty = property.FindPropertyRelative(nameof(OVRPassthroughLayer.SerializedSurfaceGeometry.meshFilter));
        var updateTransformProperty = property.FindPropertyRelative(nameof(OVRPassthroughLayer.SerializedSurfaceGeometry.updateTransform));

        using (new EditorGUI.PropertyScope(rect, label, property))
        {
            var r = rect;
            r.width /= 2;
            EditorGUI.PropertyField(r, meshFilterProperty, new GUIContent("Surface Geometry",
                "The GameObject from which to generate surface geometry."));
            r.x += r.width + 16;
            r.width -= 16;
            EditorGUI.PropertyField(r, updateTransformProperty, new GUIContent("Update Transform",
                "When enabled, updates the mesh's transform every frame. Use this if the GameObject is dynamic."));
        }
    }
}

[CustomEditor(typeof(OVRPassthroughLayer))]
public class OVRPassthroughLayerEditor : Editor {
    private readonly static string[] _selectableColorMapNames = {
        "None",
        "Color Adjustment",
        "Grayscale",
        "Grayscale To Color"
    };
    private readonly static string[] _colorMapNames = {
        "None",
        "Color Adjustment",
        "Grayscale",
        "Grayscale to color",
        "Custom"
    };
    private ColorMapEditorType[] _colorMapTypes = {
        ColorMapEditorType.None,
        ColorMapEditorType.ColorAdjustment,
        ColorMapEditorType.Grayscale,
        ColorMapEditorType.GrayscaleToColor,
        ColorMapEditorType.Custom
    };
    private SerializedProperty _projectionSurfaces;

    private SerializedProperty _propProjectionSurfaceType;
    private SerializedProperty _propOverlayType;
    private SerializedProperty _propCompositionDepth;

    private SerializedProperty _propTextureOpacity;
    private SerializedProperty _propEdgeRenderingEnabled;
    private SerializedProperty _propEdgeColor;
    private SerializedProperty _propColorMapEditorContrast;
    private SerializedProperty _propColorMapEditorBrightness;
    private SerializedProperty _propColorMapEditorPosterize;
    private SerializedProperty _propColorMapEditorGradient;
    private SerializedProperty _propColorMapEditorSaturation;


    void OnEnable()
    {
        _projectionSurfaces = serializedObject.FindProperty(nameof(OVRPassthroughLayer.serializedSurfaceGeometry));

        _propProjectionSurfaceType = serializedObject.FindProperty(nameof(OVRPassthroughLayer.projectionSurfaceType));
        _propOverlayType = serializedObject.FindProperty(nameof(OVRPassthroughLayer.overlayType));
        _propCompositionDepth = serializedObject.FindProperty(nameof(OVRPassthroughLayer.compositionDepth));
        _propTextureOpacity = serializedObject.FindProperty(nameof(OVRPassthroughLayer.textureOpacity_));
        _propEdgeRenderingEnabled = serializedObject.FindProperty(nameof(OVRPassthroughLayer.edgeRenderingEnabled_));
        _propEdgeColor = serializedObject.FindProperty(nameof(OVRPassthroughLayer.edgeColor_));
        _propColorMapEditorContrast = serializedObject.FindProperty(nameof(OVRPassthroughLayer.colorMapEditorContrast));
        _propColorMapEditorBrightness = serializedObject.FindProperty(nameof(OVRPassthroughLayer.colorMapEditorBrightness));
        _propColorMapEditorPosterize = serializedObject.FindProperty(nameof(OVRPassthroughLayer.colorMapEditorPosterize));
        _propColorMapEditorSaturation = serializedObject.FindProperty(nameof(OVRPassthroughLayer.colorMapEditorSaturation));
        _propColorMapEditorGradient = serializedObject.FindProperty(nameof(OVRPassthroughLayer.colorMapEditorGradient));

    }

    public override void OnInspectorGUI()
    {
        OVRPassthroughLayer layer = (OVRPassthroughLayer)target;

        serializedObject.Update();
        EditorGUILayout.PropertyField(_propProjectionSurfaceType,
            new GUIContent("Projection Surface", "The type of projection surface for this Passthrough layer"));

        if (layer.projectionSurfaceType == OVRPassthroughLayer.ProjectionSurfaceType.UserDefined)
        {
            EditorGUILayout.PropertyField(_projectionSurfaces, new GUIContent("Projection Surfaces"));
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Compositing", EditorStyles.boldLabel);

        EditorGUILayout.PropertyField(_propOverlayType,
            new GUIContent("Placement", "Whether this overlay should layer behind the scene or in front of it"));
        EditorGUILayout.PropertyField(_propCompositionDepth,
            new GUIContent("Composition Depth",
                "Depth value used to sort layers in the scene, smaller value appears in front"));


        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Style", EditorStyles.boldLabel);

        EditorGUILayout.Slider(_propTextureOpacity, 0, 1f, new GUIContent("Opacity"));

        EditorGUILayout.Space();

        EditorGUILayout.PropertyField(_propEdgeRenderingEnabled, new GUIContent("Edge Rendering", "Highlight salient edges in the camera images in a specific color"));
        EditorGUILayout.PropertyField(_propEdgeColor, new GUIContent("Edge Color"));

        if (serializedObject.ApplyModifiedProperties())
        {
            layer.SetStyleDirty();
        }

        layer.textureOpacity = _propTextureOpacity.floatValue;
        layer.edgeRenderingEnabled = _propEdgeRenderingEnabled.boolValue;
        layer.edgeColor = _propEdgeColor.colorValue;

        EditorGUILayout.Space();

        // Custom popup for color map type to control order, names, and visibility of types
        int colorMapTypeIndex = Array.IndexOf(_colorMapTypes, layer.colorMapEditorType);
        if (colorMapTypeIndex == -1)
        {
            Debug.LogWarning("Invalid color map type encountered");
            colorMapTypeIndex = 0;
        }
        // Dropdown list contains "Custom" only if it is currently selected.
        string[] colorMapNames = layer.colorMapEditorType == ColorMapEditorType.Custom ? _colorMapNames
            : _selectableColorMapNames;
        GUIContent[] colorMapLabels = new GUIContent[colorMapNames.Length];
        for (int i = 0; i < colorMapNames.Length; i++)
            colorMapLabels[i] = new GUIContent(colorMapNames[i]);
        bool modified = false;
        OVREditorUtil.SetupPopupField(target,
            new GUIContent("Color Control", "The type of color controls applied to this layer"), ref colorMapTypeIndex,
            colorMapLabels,
            ref modified);
        layer.colorMapEditorType = _colorMapTypes[colorMapTypeIndex];

        if (layer.colorMapEditorType == ColorMapEditorType.Grayscale
            || layer.colorMapEditorType == ColorMapEditorType.GrayscaleToColor
            || layer.colorMapEditorType == ColorMapEditorType.ColorAdjustment
        ) {
            EditorGUILayout.PropertyField(_propColorMapEditorContrast, new GUIContent("Contrast"));
            EditorGUILayout.PropertyField(_propColorMapEditorBrightness, new GUIContent("Brightness"));
        }

        if (layer.colorMapEditorType == ColorMapEditorType.Grayscale
            || layer.colorMapEditorType == ColorMapEditorType.GrayscaleToColor)
        {
            EditorGUILayout.PropertyField(_propColorMapEditorPosterize, new GUIContent("Posterize"));
        }

        if (layer.colorMapEditorType == ColorMapEditorType.ColorAdjustment)
        {
            EditorGUILayout.PropertyField(_propColorMapEditorSaturation, new GUIContent("Saturation"));
        }

        if (layer.colorMapEditorType == ColorMapEditorType.GrayscaleToColor)
        {
            EditorGUILayout.PropertyField(_propColorMapEditorGradient, new GUIContent("Colorize"));
        }

        serializedObject.ApplyModifiedProperties();
    }
}

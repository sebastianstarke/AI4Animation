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
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using ColorMapType = OVRPlugin.InsightPassthroughColorMapType;

/// <summary>
/// A layer used for passthrough.
/// </summary>
public class OVRPassthroughLayer : MonoBehaviour
{
    #region Public Interface

    /// <summary>
    /// The passthrough projection surface type: reconstructed | user defined.
    /// </summary>
    public enum ProjectionSurfaceType
    {
        Reconstructed, ///< Reconstructed surface type will render passthrough using automatic environment depth reconstruction
        UserDefined ///< UserDefined allows you to define a surface
    }

    /// <summary>
    /// The type of the surface which passthrough textures are projected on: Automatic reconstruction or user-defined geometry.
    /// This field can only be modified immediately after the component is instantiated (e.g. using `AddComponent`).
    /// Once the backing layer has been created, changes won't be reflected unless the layer is disabled and enabled again.
    /// Default is automatic reconstruction.
    /// </summary>
    public ProjectionSurfaceType projectionSurfaceType = ProjectionSurfaceType.Reconstructed;

    /// <summary>
    /// Overlay type that defines the placement of the passthrough layer to appear on top as an overlay or beneath as an underlay of the application’s main projection layer. By default, the passthrough layer appears as an overlay.
    /// </summary>
    public OVROverlay.OverlayType overlayType = OVROverlay.OverlayType.Overlay;

    /// <summary>
    /// The compositionDepth defines the order of the layers in composition. The layer with smaller compositionDepth would be composited in the front of the layer with larger compositionDepth. The default value is zero.
    /// </summary>
    public int compositionDepth = 0;

    /// <summary>
    /// Property that can hide layers when required. Should be false when present, true when hidden. By default, the value is set to false, which means the layers are present.
    /// </summary>
    public bool hidden = false;


    /// <summary>
    /// Specify whether `colorScale` and `colorOffset` should be applied to this layer. By default, the color scale and offset are not applied to the layer.
    /// </summary>
    public bool overridePerLayerColorScaleAndOffset = false;

    /// <summary>
    /// Color scale is a factor applied to the pixel color values during compositing.
    /// The four components of the vector correspond to the R, G, B, and A values, default set to `{1,1,1,1}`.
    /// </summary>
    public Vector4 colorScale = Vector4.one;

    /// <summary>
    /// Color offset is a value which gets added to the pixel color values during compositing.
    /// The four components of the vector correspond to the R, G, B, and A values, default set to `{0,0,0,0}`.
    /// </summary>
    public Vector4 colorOffset = Vector4.zero;

    /// <summary>
    /// Add a GameObject to the Insight Passthrough projection surface. This is only applicable
    /// if the projection surface type is `UserDefined`.
    /// When `updateTransform` parameter is set to `true`, OVRPassthroughLayer will update the transform
    /// of the surface mesh every frame. Otherwise only the initial transform is recorded.
    /// </summary>
    /// <param name="obj">The Gameobject you want to add to the Insight Passthrough projection surface.</param>
    /// <param name="updateTransform">Indicate if the transform should be updated every frame</param>
    public void AddSurfaceGeometry(GameObject obj, bool updateTransform = false)
    {
        if (projectionSurfaceType != ProjectionSurfaceType.UserDefined)
        {
            Debug.LogError("Passthrough layer is not configured for surface projected passthrough.");
            return;
        }

        if (surfaceGameObjects.ContainsKey(obj))
        {
            Debug.LogError("Specified GameObject has already been added as passthrough surface.");
            return;
        }

        if (obj.GetComponent<MeshFilter>() == null)
        {
            Debug.LogError("Specified GameObject does not have a mesh component.");
            return;
        }

        // Mesh and instance can't be created immediately, because the compositor layer may not have been initialized yet (layerId = 0).
        // Queue creation and attempt to do it in the update loop.
        deferredSurfaceGameObjects.Add(
            new DeferredPassthroughMeshAddition
            {
                gameObject = obj,
                updateTransform = updateTransform
            });
    }

    /// <summary>
    /// Removes a GameObject that was previously added using `AddSurfaceGeometry` from the projection surface.
    /// </summary>
    /// <param name="obj">The Gameobject to remove. </param>
    public void RemoveSurfaceGeometry(GameObject obj)
    {
        PassthroughMeshInstance passthroughMeshInstance;
        if (surfaceGameObjects.TryGetValue(obj, out passthroughMeshInstance))
        {
            if (OVRPlugin.DestroyInsightPassthroughGeometryInstance(passthroughMeshInstance.instanceHandle) &&
                    OVRPlugin.DestroyInsightTriangleMesh(passthroughMeshInstance.meshHandle))
            {
                surfaceGameObjects.Remove(obj);
            }
            else
            {
                Debug.LogError("GameObject could not be removed from passthrough surface.");
            }
        }
        else
        {
            int count = deferredSurfaceGameObjects.RemoveAll(x => x.gameObject == obj);
            if (count == 0)
            {
                Debug.LogError("Specified GameObject has not been added as passthrough surface.");
            }
        }
    }

    /// <summary>
    /// Checks if the given gameobject is a surface geometry (If called with AddSurfaceGeometry).
    /// </summary>
    /// <returns> True if the gameobject is a surface geometry. </returns>
    public bool IsSurfaceGeometry(GameObject obj)
    {
        return surfaceGameObjects.ContainsKey(obj) || deferredSurfaceGameObjects.Exists(x => x.gameObject == obj);
    }

    /// <summary>
    /// Float that defines the passthrough texture opacity.
    /// </summary>
    public float textureOpacity
    {
        get
        {
            return textureOpacity_;
        }
        set
        {
            if (value != textureOpacity_)
            {
                textureOpacity_ = value;
                styleDirty = true;
            }
        }
    }

    /// <summary>
    /// Enable or disable the Edge rendering.
    /// Use this flag to enable or disable the edge rendering but retain the previously selected color (incl. alpha)
    /// in the UI when it is disabled.
    /// </summary>
    public bool edgeRenderingEnabled
    {
        get
        {
            return edgeRenderingEnabled_;
        }
        set
        {
            if (value != edgeRenderingEnabled_)
            {
                edgeRenderingEnabled_ = value;
                styleDirty = true;
            }
        }
    }

    /// <summary>
    /// Color for the edge rendering.
    /// </summary>
    public Color edgeColor
    {
        get
        {
            return edgeColor_;
        }
        set
        {
            if (value != edgeColor_)
            {
                edgeColor_ = value;
                styleDirty = true;
            }
        }
    }

    /// <summary>
    /// This color map method allows to recolor the grayscale camera images by specifying a color lookup table.
    /// Scripts should call the designated methods to set a color map. The fields and properties
    /// are only intended for the inspector UI.
    /// </summary>
    /// <param name="values">The color map as an array of 256 color values to map each grayscale input to a color.</param>
    public void SetColorMap(Color[] values)
    {
        if (values.Length != 256)
            throw new ArgumentException("Must provide exactly 256 colors");

        colorMapType = ColorMapType.MonoToRgba;
        colorMapEditorType = ColorMapEditorType.Custom;
        AllocateColorMapData();
        for (int i = 0; i < 256; i++)
        {
            WriteColorToColorMap(i, ref values[i]);
        }

        styleDirty = true;
    }


    /// <summary>
    /// This method allows to generate (and apply) a color map from the set of controls which is also available in
    /// inspector.
    /// </summary>
    /// <param name="contrast">The contrast value. Range from -1 (minimum) to 1 (maximum). </param>
    /// <param name="brightness">The brightness value. Range from 0 (minimum) to 1 (maximum). </param>
    /// <param name="posterize">The posterize value. Range from 0 to 1, where 0 = no posterization (no effect), 1 = reduce to two colors. </param>
    /// <param name="gradient">The gradient will be evaluated from 0 (no intensity) to 1 (maximum intensity).
    /// 	This parameter only has an effect if `colorMapType` is `GrayscaleToColor`.</param>
    /// <param name="colorMapType">Type of color map which should be generated. Supported values: `Grayscale` and `GrayscaleToColor`.</param>
    public void SetColorMapControls(
        float contrast,
        float brightness = 0.0f,
        float posterize = 0.0f,
        Gradient gradient = null,
        ColorMapEditorType colorMapType = ColorMapEditorType.GrayscaleToColor)
    {
        if (!(colorMapType == ColorMapEditorType.Grayscale || colorMapType == ColorMapEditorType.GrayscaleToColor)) {
            Debug.LogError("Unsupported color map type specified");
            return;
        }

        colorMapEditorType = colorMapType;
        colorMapEditorContrast = contrast;
        colorMapEditorBrightness = brightness;
        colorMapEditorPosterize = posterize;

        if (colorMapType == ColorMapEditorType.GrayscaleToColor) {
            if (gradient != null)
            {
                colorMapEditorGradient = gradient;
            }
            else if (!colorMapEditorGradient.Equals(colorMapNeutralGradient))
            {
                // Leave gradient untouched if it's already neutral to avoid unnecessary memory allocations.
                colorMapEditorGradient = CreateNeutralColorMapGradient();
            }
        } else if (gradient != null) {
            Debug.LogWarning("Gradient parameter is ignored for color map types other than GrayscaleToColor");
        }
    }

    /// <summary>
    /// This method allows to specify the color map as an array of 256 8-bit intensity values.
    /// Use this to map each grayscale input value to a grayscale output value.
    /// </summary>
    /// <param name="values">Array of 256 8-bit values.</param>
    public void SetColorMapMonochromatic(byte[] values)
    {
        if (values.Length != 256)
            throw new ArgumentException("Must provide exactly 256 values");

        colorMapType = ColorMapType.MonoToMono;
        colorMapEditorType = ColorMapEditorType.Custom;
        AllocateColorMapData();
        Buffer.BlockCopy(values, 0, colorMapData, 0, 256);

        styleDirty = true;
    }

    /// <summary>
    /// This method allows to configure brightness and contrast adjustment for Passthrough images.
    /// </summary>
    /// <param name="brightness">Modify the brightness of Passthrough. Valid range: [-1, 1]. A
    ///   value of 0 means that brightness is left unchanged.</param>
    /// <param name="contrast">Modify the contrast of Passthrough. Valid range: [-1, 1]. A value of 0
    ///   means that contrast is left unchanged.</param>
    /// <param name="saturation">Modify the saturation of Passthrough. Valid range: [-1, 1]. A value
    ///   of 0 means that saturation is left unchanged.</param>
    public void SetBrightnessContrastSaturation(float brightness = 0.0f, float contrast = 0.0f, float saturation = 0.0f)
    {
        colorMapType = ColorMapType.BrightnessContrastSaturation;
        colorMapEditorType = ColorMapEditorType.ColorAdjustment;
        AllocateColorMapData();
        colorMapEditorBrightness = brightness;
        colorMapEditorContrast = contrast;
        colorMapEditorSaturation = saturation;

        UpdateColorMapFromControls();
    }


    /// <summary>
    /// Disables color mapping. Use this to remove any effects.
    /// </summary>
    public void DisableColorMap()
    {
        colorMapEditorType = ColorMapEditorType.None;
    }

    #endregion


    #region Editor Interface
    /// <summary>
    /// Unity editor enumerator to provide a dropdown in the inspector.
    /// </summary>
    public enum ColorMapEditorType
    {
        None = 0, ///< No color map is applied
        GrayscaleToColor = 1, ///< Map input color to an RGB color, optionally with brightness/constrast adjustment or posterization applied.
        Controls = GrayscaleToColor, ///< Deprecated - use GrayscaleToColor instead.
        Custom = 2, ///< Color map is specified using one of the class setters.
        Grayscale = 3, ///< Map input color to a grayscale color, optionally with brightness/constrast adjustment or posterization applied.
        ColorAdjustment = 4 ///< Adjust brightness and contrast
    }

    [SerializeField]
    internal ColorMapEditorType colorMapEditorType_ = ColorMapEditorType.None;
    /// <summary>
    /// Editor attribute to get or set the selection in the inspector.
    /// Using this selection will update the `colorMapType` and `colorMapData` if needed.
    /// </summary>
    public ColorMapEditorType colorMapEditorType
    {
        get
        {
            return colorMapEditorType_;
        }
        set
        {
            if (value != colorMapEditorType_)
            {
                colorMapEditorType_ = value;

                // Update colorMapType and colorMapData to match new editor selection
                switch (value)
                {
                    case ColorMapEditorType.None:
                        colorMapType = ColorMapType.None;
                        DeallocateColorMapData();
                        styleDirty = true;
                        break;
                    case ColorMapEditorType.Grayscale:
                        colorMapType = ColorMapType.MonoToMono;
                        UpdateColorMapFromControls(true);
                        break;
                    case ColorMapEditorType.GrayscaleToColor:
                        colorMapType = ColorMapType.MonoToRgba;
                        UpdateColorMapFromControls(true);
                        break;
                    case ColorMapEditorType.ColorAdjustment:
                        colorMapType = ColorMapType.BrightnessContrastSaturation;
                        UpdateColorMapFromControls(true);
                        break;
                    case ColorMapEditorType.Custom:
                        // no-op
                        break;
                }
            }
        }
    }

    /// <summary>
    /// This field is not intended for public scripting. Use `SetColorMapControls()` instead.
    /// </summary>
    public Gradient colorMapEditorGradient = CreateNeutralColorMapGradient();

    // Keep a private copy of the gradient value. Every frame, it is compared against the public one in UpdateColorMapFromControls() and updated if necessary.
    private Gradient colorMapEditorGradientOld = new Gradient();

    /// <summary>
    /// This field is not intended for public scripting. Use `SetBrightnessContrastSaturation()` or `SetColorMapControls()` instead.
    /// </summary>
    [Range(-1f,1f)]
    public float colorMapEditorContrast;
    // Keep a private copy of the contrast value. Every frame, it is compared against the public one in UpdateColorMapFromControls() and updated if necessary.
    private float colorMapEditorContrast_ = 0;

    /// <summary>
    /// This field is not intended for public scripting. Use `SetBrightnessContrastSaturation()` or `SetColorMapControls()` instead.
    /// </summary>
    [Range(-1f,1f)]
    public float colorMapEditorBrightness;
    // Keep a private copy of the brightness value. Every frame, it is compared against the public one in UpdateColorMapFromControls() and updated if necessary.
    private float colorMapEditorBrightness_ = 0;

    /// <summary>
    /// This field is not intended for public scripting. Use `SetColorMapControls()` instead.
    /// </summary>
    [Range(0f,1f)]
    public float colorMapEditorPosterize;
    // Keep a private copy of the posterize value. Every frame, it is compared against the public one in UpdateColorMapFromControls() and updated if necessary.
    private float colorMapEditorPosterize_ = 0;

    /// <summary>
    /// This field is not intended for public scripting. Use `SetBrightnessContrastSaturation()` instead.
    /// </summary>
    [Range(-1f,1f)]
    public float colorMapEditorSaturation;
    // Keep a private copy of the saturation value. Every frame, it is compared against the public one in UpdateColorMapFromControls() and updated if necessary.
    private float colorMapEditorSaturation_ = 0;

    /// <summary>
    /// This method is required for internal use only.
    /// </summary>
    public void SetStyleDirty()
    {
        styleDirty = true;
    }

    #endregion

    #region Internal Methods
    private void AddDeferredSurfaceGeometries()
    {
        for (int i = 0; i < deferredSurfaceGameObjects.Count; ++i)
        {
            var entry = deferredSurfaceGameObjects[i];
            bool entryIsPassthroughObject = false;
            if (surfaceGameObjects.ContainsKey(entry.gameObject))
            {
                entryIsPassthroughObject = true;
            }
            else
            {
                if (CreateAndAddMesh(entry.gameObject, out var meshHandle, out var instanceHandle, out var localToWorld))
                {
                    surfaceGameObjects.Add(entry.gameObject, new PassthroughMeshInstance
                    {
                        meshHandle = meshHandle,
                        instanceHandle = instanceHandle,
                        updateTransform = entry.updateTransform,
                        localToWorld = localToWorld,
                    });
                    entryIsPassthroughObject = true;
                }
                else
                {
                    Debug.LogWarning("Failed to create internal resources for GameObject added to passthrough surface.");
                }
            }

            if (entryIsPassthroughObject)
            {
                deferredSurfaceGameObjects.RemoveAt(i--);
            }
        }
    }

    private Matrix4x4 GetTransformMatrixForPassthroughSurfaceObject(Matrix4x4 worldFromObj)
    {
        using var profile = new OVRProfilerScope(nameof(GetTransformMatrixForPassthroughSurfaceObject));

        if (!cameraRigInitialized)
        {
            cameraRig = OVRManager.instance.GetComponentInParent<OVRCameraRig>();
            cameraRigInitialized = true;
        }

        Matrix4x4 trackingSpaceFromWorld = (cameraRig != null) ?
            cameraRig.trackingSpace.worldToLocalMatrix :
            Matrix4x4.identity;

        // Use model matrix to switch from left-handed coordinate system (Unity)
        // to right-handed (Open GL/Passthrough API): reverse z-axis
        Matrix4x4 rightHandedFromLeftHanded = Matrix4x4.Scale(new Vector3(1, 1, -1));
        return rightHandedFromLeftHanded * trackingSpaceFromWorld * worldFromObj;
    }

    private bool CreateAndAddMesh(
        GameObject obj,
        out ulong meshHandle,
        out ulong instanceHandle,
        out Matrix4x4 localToWorld)
    {
        Debug.Assert(passthroughOverlay != null);
        Debug.Assert(passthroughOverlay.layerId > 0);
        meshHandle = 0;
        instanceHandle = 0;
        localToWorld = obj.transform.localToWorldMatrix;

        MeshFilter meshFilter = obj.GetComponent<MeshFilter>();
        if (meshFilter == null)
        {
            Debug.LogError("Passthrough surface GameObject does not have a mesh component.");
            return false;
        }

        Mesh mesh = meshFilter.sharedMesh;

        // TODO: evaluate using GetNativeVertexBufferPtr() instead to avoid copy
        Vector3[] vertices = mesh.vertices;
        int[] triangles = mesh.triangles;
        Matrix4x4 T_worldInsight_model = GetTransformMatrixForPassthroughSurfaceObject(localToWorld);

        if (!OVRPlugin.CreateInsightTriangleMesh(passthroughOverlay.layerId, vertices, triangles, out meshHandle))
        {
            Debug.LogWarning("Failed to create triangle mesh handle.");
            return false;
        }

        if (!OVRPlugin.AddInsightPassthroughSurfaceGeometry(passthroughOverlay.layerId, meshHandle, T_worldInsight_model, out instanceHandle))
        {
            Debug.LogWarning("Failed to add mesh to passthrough surface.");
            return false;
        }

        return true;
    }

    private void DestroySurfaceGeometries(bool addBackToDeferredQueue = false)
    {
        foreach (KeyValuePair<GameObject, PassthroughMeshInstance> el in surfaceGameObjects)
        {
            if (el.Value.meshHandle != 0)
            {
                OVRPlugin.DestroyInsightPassthroughGeometryInstance(el.Value.instanceHandle);
                OVRPlugin.DestroyInsightTriangleMesh(el.Value.meshHandle);

                // When DestroySurfaceGeometries is called from OnDisable, we want to keep track of the existing
                // surface geometries so we can add them back when the script gets enabled again. We simply reinsert
                // them into deferredSurfaceGameObjects for that purpose.
                if (addBackToDeferredQueue)
                {
                    deferredSurfaceGameObjects.Add(
                        new DeferredPassthroughMeshAddition
                        {
                            gameObject = el.Key,
                            updateTransform = el.Value.updateTransform
                        });
                }
            }
        }
        surfaceGameObjects.Clear();
    }

    private void UpdateSurfaceGeometryTransforms()
    {
        using var profile = new OVRProfilerScope(nameof(UpdateSurfaceGeometryTransforms));

        // Iterate through mesh instances and see if transforms need to be updated
        foreach (var kvp in surfaceGameObjects)
        {
            var instanceHandle = kvp.Value.instanceHandle;
            if (instanceHandle == 0) continue;

            var localToWorld = kvp.Value.updateTransform
                ? kvp.Key.transform.localToWorldMatrix
                : kvp.Value.localToWorld;

            UpdateSurfaceGeometryTransform(instanceHandle, localToWorld);
        }
    }

    private void UpdateSurfaceGeometryTransform(ulong instanceHandle, Matrix4x4 localToWorld)
    {
        var worldInsightModel = GetTransformMatrixForPassthroughSurfaceObject(localToWorld);
        using (new OVRProfilerScope(nameof(OVRPlugin.UpdateInsightPassthroughGeometryTransform)))
        {
            if (!OVRPlugin.UpdateInsightPassthroughGeometryTransform(instanceHandle, worldInsightModel))
            {
                Debug.LogWarning("Failed to update a transform of a passthrough surface");
            }
        }
    }

    private void AllocateColorMapData(uint size = 4096)
    {
        if (colorMapData != null && size != colorMapData.Length) {
            DeallocateColorMapData();
        }

        if (colorMapData == null)
        {
            colorMapData = new byte[size];
            if (colorMapDataHandle.IsAllocated)
            {
                Debug.LogWarning("Passthrough color map data handle is not expected to be allocated at time of buffer allocation");
            }
            colorMapDataHandle = GCHandle.Alloc(colorMapData, GCHandleType.Pinned);

            tmpColorMapData = new byte[256];
        }
    }

    // Ensure that Passthrough color map data is unpinned and freed.
    private void DeallocateColorMapData()
    {
        if (colorMapData != null)
        {
            if (!colorMapDataHandle.IsAllocated)
            {
                Debug.LogWarning("Passthrough color map data handle is expected to be allocated at time of buffer deallocation");
            }
            else
            {
                colorMapDataHandle.Free();
            }
            colorMapData = null;
            tmpColorMapData = null;
        }
    }

    // Returns a gradient from black to white.
    private static Gradient CreateNeutralColorMapGradient()
    {
        return new Gradient()
        {
            colorKeys = new GradientColorKey[2] {
                new GradientColorKey(new Color(0, 0, 0), 0),
                new GradientColorKey(new Color(1, 1, 1), 1)
            },
            alphaKeys = new GradientAlphaKey[2] {
                new GradientAlphaKey(1, 0),
                new GradientAlphaKey(1, 1)
            }
        };
    }

    private bool HasControlsBasedColorMap()
    {
        return colorMapEditorType == ColorMapEditorType.Grayscale
                || colorMapEditorType == ColorMapEditorType.ColorAdjustment
                || colorMapEditorType == ColorMapEditorType.GrayscaleToColor;
    }

    private void UpdateColorMapFromControls(bool forceUpdate = false)
    {
        bool parametersChanged = colorMapEditorBrightness_ != colorMapEditorBrightness
                || colorMapEditorContrast_ != colorMapEditorContrast
                || colorMapEditorPosterize_ != colorMapEditorPosterize
                || colorMapEditorSaturation_ != colorMapEditorSaturation;
        bool gradientNeedsUpdate = colorMapEditorType == ColorMapEditorType.GrayscaleToColor
            && !colorMapEditorGradient.Equals(colorMapEditorGradientOld);

        if (!(HasControlsBasedColorMap() && parametersChanged || gradientNeedsUpdate || forceUpdate))
            return;

        AllocateColorMapData();

        colorMapEditorGradientOld.CopyFrom(colorMapEditorGradient);
        colorMapEditorBrightness_ = colorMapEditorBrightness;
        colorMapEditorContrast_ = colorMapEditorContrast;
        colorMapEditorPosterize_ = colorMapEditorPosterize;
        colorMapEditorSaturation_ = colorMapEditorSaturation;

        switch (colorMapEditorType)
        {
            case ColorMapEditorType.Grayscale:
                computeBrightnessContrastPosterizeMap(colorMapData, colorMapEditorBrightness, colorMapEditorContrast, colorMapEditorPosterize);
                styleDirty = true;
                break;
            case ColorMapEditorType.GrayscaleToColor:
                computeBrightnessContrastPosterizeMap(tmpColorMapData, colorMapEditorBrightness, colorMapEditorContrast, colorMapEditorPosterize);
                for (int i = 0; i < 256; i++)
                {
                    Color color = colorMapEditorGradient.Evaluate(tmpColorMapData[i]  / 255.0f);
                    WriteColorToColorMap(i, ref color);
                }
                styleDirty = true;
                break;
            case ColorMapEditorType.ColorAdjustment:
                WriteBrightnessContrastSaturationColorMap(colorMapEditorBrightness_, colorMapEditorContrast_, colorMapEditorSaturation_);
                styleDirty = true;
                break;
        }
    }

    static private void computeBrightnessContrastPosterizeMap(byte[] result, float brightness, float contrast, float posterize)
    {
        for (int i = 0; i < 256; i++)
        {
            // Apply contrast, brightness and posterization on the grayscale value
            float value = i / 255.0f;
            // Constrast and brightness
            float contrastFactor = contrast + 1; // UI runs from -1 to 1
            value = (value - 0.5f) * contrastFactor + 0.5f + brightness;

            // Posterization
            if (posterize > 0.0f)
            {
                // The posterization slider feels more useful if the progression is exponential. The function is emprically tuned.
                const float posterizationBase = 50.0f;
                float quantization = (Mathf.Pow(posterizationBase, posterize) - 1.0f) / (posterizationBase - 1.0f);
                value = Mathf.Round(value / quantization) * quantization;
            }

            result[i] = (byte)(Mathf.Min(Mathf.Max(value, 0.0f), 1.0f) * 255.0f);
        }
    }

    // Write a single color value to the Passthrough color map at the given position.
    private void WriteColorToColorMap(int colorIndex, ref Color color)
    {
        for (int c = 0; c < 4; c++)
        {
            byte[] bytes = BitConverter.GetBytes(color[c]);
            Buffer.BlockCopy(bytes, 0, colorMapData, colorIndex * 16 + c * 4, 4);
        }
    }


    private void WriteFloatToColorMap(int index, float value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Buffer.BlockCopy(bytes, 0, colorMapData, index * sizeof(float), sizeof(float));
    }

    private void WriteBrightnessContrastSaturationColorMap(float brightness, float contrast, float saturation)
    {
        // Brightness: input is in range [-1, 1], output [0, 100]
        WriteFloatToColorMap(0, brightness * 100.0f);

        // Contrast: input is in range [-1, 1], output [0, 2]
        WriteFloatToColorMap(1, contrast + 1.0f);

        // Saturation: input is in range [-1, 1], output [0, 2]
        WriteFloatToColorMap(2, saturation + 1.0f);
    }

    private void SyncToOverlay()
    {
        Debug.Assert(passthroughOverlay != null);

        passthroughOverlay.currentOverlayType = overlayType;
        passthroughOverlay.compositionDepth = compositionDepth;
        passthroughOverlay.hidden = hidden;
        passthroughOverlay.overridePerLayerColorScaleAndOffset = overridePerLayerColorScaleAndOffset;
        passthroughOverlay.colorScale = colorScale;
        passthroughOverlay.colorOffset = colorOffset;

        if (passthroughOverlay.currentOverlayShape != overlayShape)
        {
            if (passthroughOverlay.layerId > 0)
            {
                Debug.LogWarning("Change to projectionSurfaceType won't take effect until the layer goes through a disable/enable cycle. ");
            }

            if (projectionSurfaceType == ProjectionSurfaceType.Reconstructed)
            {
                // Ensure there are no custom surface geometries when switching to reconstruction passthrough.
                Debug.Log("Removing user defined surface geometries");
                DestroySurfaceGeometries(false);
            }

            passthroughOverlay.currentOverlayShape = overlayShape;
        }

        // Disable the overlay when passthrough is disabled as a whole so the layer doesn't get submitted.
        // Both the desired (`isInsightPassthroughEnabled`) and the actual (IsInsightPassthroughInitialized()) PT
        // initialization state are taken into account s.t. the overlay gets disabled as soon as PT is flagged to be
        // disabled, and enabled only when PT is up and running again.
        passthroughOverlay.enabled = OVRManager.instance != null &&
            OVRManager.instance.isInsightPassthroughEnabled &&
            OVRManager.IsInsightPassthroughInitialized();
    }

    #endregion

    #region Internal Fields/Properties
    private OVRCameraRig cameraRig;
    private bool cameraRigInitialized = false;
    private GameObject auxGameObject;
    private OVROverlay passthroughOverlay;

    // Each GameObjects requires a MrTriangleMesh and a MrPassthroughGeometryInstance handle.
    // The structure also keeps a flag for whether transform updates should be tracked.
    private struct PassthroughMeshInstance
    {
        public ulong meshHandle;
        public ulong instanceHandle;
        public bool updateTransform;
        public Matrix4x4 localToWorld;
    }

    [Serializable]
    internal struct SerializedSurfaceGeometry
    {
        public MeshFilter meshFilter;
        public bool updateTransform;
    }

    // A structure for tracking a deferred addition of a game object to the projection surface
    private struct DeferredPassthroughMeshAddition
    {
        public GameObject gameObject;
        public bool updateTransform;
    }

    // GameObjects which are in use as Insight Passthrough projection surface.
    private Dictionary<GameObject, PassthroughMeshInstance> surfaceGameObjects =
            new Dictionary<GameObject, PassthroughMeshInstance>();

    // GameObjects which are pending addition to the Insight Passthrough projection surfaces.
    private List<DeferredPassthroughMeshAddition> deferredSurfaceGameObjects =
            new List<DeferredPassthroughMeshAddition>();

    [SerializeField, HideInInspector]
    internal List<SerializedSurfaceGeometry> serializedSurfaceGeometry =
        new List<SerializedSurfaceGeometry>();

    [SerializeField]
    internal float textureOpacity_ = 1;

    [SerializeField]
    internal bool edgeRenderingEnabled_ = false;

    [SerializeField]
    internal Color edgeColor_ = new Color(1, 1, 1, 1);

    // Internal fields which store the color map values that will be relayed to the Passthrough API in the next update.
    [SerializeField]
    private ColorMapType colorMapType = ColorMapType.None;

    // Passthrough color map data gets allocated and deallocated on demand.
    private byte[] colorMapData = null;

    // Buffer used to store intermediate results for color map computations.
    private byte[] tmpColorMapData = null;

    // Passthrough color map data gets pinned in the GC on allocation so it can be passed to the native side safely.
    // In remains pinned for its lifecycle to avoid pinning per frame and the resulting memory allocation and GC pressure.
    private GCHandle colorMapDataHandle;


    // Flag which indicates whether the style values have changed since the last update in the Passthrough API.
    // It is set to `true` initially to ensure that the local default values are applied in the Passthrough API.
    private bool styleDirty = true;

    // Keep a copy of a neutral gradient ready for comparison.
    static readonly private Gradient colorMapNeutralGradient = CreateNeutralColorMapGradient();

    // Overlay shape derived from `projectionSurfaceType`.
    private OVROverlay.OverlayShape overlayShape
    {
        get
        {
            return projectionSurfaceType == ProjectionSurfaceType.UserDefined ?
                OVROverlay.OverlayShape.SurfaceProjectedPassthrough :
                OVROverlay.OverlayShape.ReconstructionPassthrough;
        }
    }
    #endregion

    #region Unity Messages

    void Awake()
    {
        foreach (var surfaceGeometry in serializedSurfaceGeometry)
        {
            if (surfaceGeometry.meshFilter == null) continue;

            deferredSurfaceGameObjects.Add(new DeferredPassthroughMeshAddition
            {
                gameObject = surfaceGeometry.meshFilter.gameObject,
                updateTransform = surfaceGeometry.updateTransform
            });
        }
    }

    void Update()
    {
        SyncToOverlay();
    }

    void LateUpdate()
    {
        if (hidden) return;

        Debug.Assert(passthroughOverlay != null);

        // This LateUpdate() should be called after passthroughOverlay's LateUpdate() such that the layerId has
        // become available at this point. This is achieved by setting the execution order of this script to a value
        // past the default time (in .meta).

        if (passthroughOverlay.layerId <= 0)
        {
            // Layer not initialized yet
            return;
        }

        if (projectionSurfaceType == ProjectionSurfaceType.UserDefined)
        {
            // Update the poses before adding new items to avoid redundant calls.
            UpdateSurfaceGeometryTransforms();

            // Delayed additon of passthrough surface geometries.
            AddDeferredSurfaceGeometries();
        }

        // Update passthrough color map with gradient if it was changed in the inspector.
        UpdateColorMapFromControls();

        // Passthrough style updates are buffered and committed to the API atomically here.
        if (styleDirty)
        {
            OVRPlugin.InsightPassthroughStyle style;
            style.Flags = OVRPlugin.InsightPassthroughStyleFlags.HasTextureOpacityFactor |
                OVRPlugin.InsightPassthroughStyleFlags.HasEdgeColor |
                OVRPlugin.InsightPassthroughStyleFlags.HasTextureColorMap;

            style.TextureOpacityFactor = textureOpacity;

            style.EdgeColor = edgeRenderingEnabled ? edgeColor.ToColorf() : new OVRPlugin.Colorf { r = 0, g = 0, b = 0, a = 0 };

            style.TextureColorMapType = colorMapType;
            style.TextureColorMapData = IntPtr.Zero;
            style.TextureColorMapDataSize = 0;

            if (style.TextureColorMapType != ColorMapType.None && colorMapData == null)
            {
                Debug.LogError("Color map not allocated");
                style.TextureColorMapType = ColorMapType.None;
            }

            if (style.TextureColorMapType != ColorMapType.None)
            {
                if (!colorMapDataHandle.IsAllocated)
                {
                    Debug.LogError("Passthrough color map enabled but data isn't pinned");
                }
                else
                {
                    style.TextureColorMapData = colorMapDataHandle.AddrOfPinnedObject();
                    switch (style.TextureColorMapType)
                    {
                        case ColorMapType.MonoToRgba:
                            style.TextureColorMapDataSize = 256 * 4 * 4; // 256 * sizeof(MrColor4f)
                            break;
                        case ColorMapType.MonoToMono:
                            style.TextureColorMapDataSize = 256;
                            break;
                        case ColorMapType.BrightnessContrastSaturation:
                            style.TextureColorMapDataSize = 3 * sizeof(float);
                            break;
                        default:
                            Debug.LogError("Unexpected texture color map type");
                            break;
                    }
                }
            }

            OVRPlugin.SetInsightPassthroughStyle(passthroughOverlay.layerId, style);

            styleDirty = false;
        }
    }

    void OnEnable()
    {
        Debug.Assert(auxGameObject == null);
        Debug.Assert(passthroughOverlay == null);

        // Create auxiliary GameObject which contains the OVROverlay component for the proxy layer (and possibly other
        // auxiliary layers in the future).
        auxGameObject = new GameObject("OVRPassthroughLayer auxiliary GameObject");

        // Auxiliary GameObject must be a child of the current GameObject s.t. it survives if `DontDestroyOnLoad` is
        // called on the current GameObject.
        auxGameObject.transform.parent = this.transform;

        // Add OVROverlay component for the passthrough proxy layer.
        passthroughOverlay = auxGameObject.AddComponent<OVROverlay>();
        passthroughOverlay.currentOverlayShape = overlayShape;
        SyncToOverlay();

        // Surface geometries have been moved to the deferred additions queue in OnDisable() and will be re-added
        // in LateUpdate().

        if (HasControlsBasedColorMap())
        {
            // Compute initial color map from controls
            UpdateColorMapFromControls(true);
        }

        // Flag style to be re-applied in LateUpdate()
        styleDirty = true;
    }

    void OnDisable()
    {
        if (OVRManager.loadedXRDevice == OVRManager.XRDevice.Oculus)
        {
            DestroySurfaceGeometries(true);
        }

        if (auxGameObject != null)
        {
            Debug.Assert(passthroughOverlay != null);
            Destroy(auxGameObject);
            auxGameObject = null;
            passthroughOverlay = null;
        }
    }

    void OnDestroy()
    {
        DestroySurfaceGeometries();
    }
#endregion
}

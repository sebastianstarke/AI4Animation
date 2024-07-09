//
// Magic Mirror Pro - Recursive Edition
// (c) 2018 Digital Ruby, LLC
// Source code may be used for personal or commercial projects.
// Source code may NOT be redistributed or sold.
// 

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DigitalRuby.MagicMirror
{
    [ExecuteInEditMode]
    public class MagicMirrorScript : MonoBehaviour
    {
        public enum AntiAliasingSampleCount
        {
            None = 1,
            Two = 2,
            Four = 4,
            Eight = 8
        };

        #region Public properties

        [Tooltip("Renderer to draw reflection in")]
        public Renderer ReflectRenderer;

        [Tooltip("What layers to reflect")]
        public LayerMask ReflectionMask = -1;

        [Tooltip("Reflection mask for recursion. Set to 0 to match the ReflectionMask property.")]
        public LayerMask ReflectionMaskRecursion = 0;

        [Tooltip("Whether to reflect the skybox")]
        public bool ReflectSkybox;

        [Tooltip("Reflection texture name for shaders to use")]
        public string ReflectionSamplerName = "_ReflectionTex";
        private string ReflectionSamplerName2;

        [Tooltip("Maximum per pixel lights in reflection")]
        [Range(0, 128)]
        public int MaximumPerPixelLightsToReflect = 8;

        [Tooltip("Set to greater than 0 to set anti-aliasing samples on the render texture. Ignored for non-forward rendering paths.")]
        public AntiAliasingSampleCount AntiAliasingSamples;

        [Tooltip("Near clip plane offset for reflection")]
        public float ClipPlaneOffset = 0.07f;

        [Tooltip("Render texture size. Based on aspect ratio this will use this size as the width or height.")]
        [Range(64, 4096)]
        public int RenderTextureSize = 1024;

        [Tooltip("The reflection camera render path. Set to 'UsePlayerSettings' to take on the observing camera rendering path. DO NOT CHANGE AT RUNTIME.")]
        public RenderingPath ReflectionCameraRenderingPath = RenderingPath.UsePlayerSettings;

        [Tooltip("Whether normal is forward. True for quads, false for planes (up)")]
        public bool NormalIsForward = true;

        [Tooltip("Aspect ratio (width/height) for reflection camera, 0 for default.")]
        [Range(0.0f, 10.0f)]
        public float AspectRatio = 0.0f;

        [Tooltip("Field of view for reflection camera, 0 for default.")]
        [Range(0.0f, 360.0f)]
        public float FieldOfView = 0.0f;

        [Tooltip("Near plane for reflection camera, 0 for default.")]
        public float NearPlane = 0.0f;

        [Tooltip("Far plane for reflection camera, 0 for default.")]
        public float FarPlane = 0.0f;

        [Tooltip("Recursion limit. Reflections will render off each other up to this many times. Be careful for performance.")]
        [Range(0, 10)]
        public int RecursionLimit = 0;

        [Tooltip("Reduce render texture size as recursion increases, formula = Mathf.Pow(RecursionRenderTextureSizeReducerPower, recursionLevel) * RenderTextureSize.")]
        [Range(0.1f, 1.0f)]
        public float RecursionRenderTextureSizeReducerPower = 0.75f;

        [Tooltip("Render texture format for reflection")]
        public RenderTextureFormat RenderTextureFormat = RenderTextureFormat.ARGB32;

        [Tooltip("Stereo separation multiplier, use this if objects are not scaling exactly the way you want.")]
        [Range(0.0f, 1.0f)]
        public float StereoSeparationMultiplier = 1.0f;

        /// <summary>
        /// The current recursion level of mirrors being rendered
        /// </summary>
        public static int CurrentRecursionLevel { get; private set; }
        #endregion Public properties

        private const string mirrorRecursionLimitKeyword = "MIRROR_RECURSION_LIMIT";

        private readonly List<ReflectionCameraInfo> currentCameras = new List<ReflectionCameraInfo>();
        private readonly List<ReflectionCameraInfo> cameraCache = new List<ReflectionCameraInfo>();
        private readonly List<KeyValuePair<RenderTexture, RenderTexture>> currentRenderTextures = new List<KeyValuePair<RenderTexture, RenderTexture>>();
        private readonly Dictionary<Camera, List<KeyValuePair<RenderTexture, StereoTargetEyeMask>>> sourceCamerasToRenderTextures = new Dictionary<Camera, List<KeyValuePair<RenderTexture, StereoTargetEyeMask>>>();
        private static readonly float[] cullDistances = new float[32];

        // prevent too many renders
        private static int renderCount;
        private const int maxRenderCount = 100;
        private const int waterLayerInverse = -17;

        private bool initialized;

        /// <summary>
        /// Information about a camera reflection
        /// </summary>
        public class ReflectionCameraInfo
        {
            /// <summary>
            /// The observing camera
            /// </summary>
            public Camera SourceCamera;

            /// <summary>
            /// The reflection camera
            /// </summary>
            public Camera ReflectionCamera;

            /// <summary>
            /// Whether the source camera is a reflection camera
            /// </summary>
            public bool SourceCameraIsReflection;

            /// <summary>
            /// Target render texture
            /// </summary>
            public RenderTexture TargetTexture;

            /// <summary>
            /// Second target render texture, only needed if VR is enabled. This is for the right eye.
            /// Unity does not provide a way to call camera.Render() with single pass stereo,
            /// each eye must be rendered to a separate render texture regardless of VR settings.
            /// </summary>
            public RenderTexture TargetTexture2;
        }

        public ReflectionCameraInfo QueueReflection(Camera sourceCamera)
        {
            bool isReflection;
            if (ShouldIgnoreCamera(sourceCamera, out isReflection))
            {
                return null;
            }
            ReflectionCameraInfo cam = CreateReflectionCamera(sourceCamera, isReflection);
            RenderReflectionCamera(cam);
            return cam;
        }

        public Camera CameraRenderingReflection(Camera sourceCamera)
        {
            for (int i = 0; i < currentCameras.Count; i++)
            {
                if (currentCameras[i].SourceCamera == sourceCamera)
                {
                    return currentCameras[i].ReflectionCamera;
                }
            }
            return null;
        }

        /// <summary>
        /// Determines whether a camera is a reflection camera
        /// </summary>
        /// <param name="cam">Camera</param>
        /// <param name="camName">Receives camera name for re-use later</param>
        /// <returns>True if cam is a reflection camera, false otherwise</returns>
        public static bool CameraIsReflection(Camera cam, out string camName)
        {
            camName = cam.name;
            return camName.IndexOf("water", StringComparison.OrdinalIgnoreCase) >= 0 ||
                camName.IndexOf("refl", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        /// <summary>
        /// Render a reflection camera.
        /// </summary>
        /// <param name="info">Camera info</param>
        /// <param name="reflectionTransform">Reflection transform (parent of reflection camera)</param>
        /// <param name="reflectionNormal">Reflection normal vector (up for planes, forward for quads)</param>
        /// <param name="clipPlaneOffset">Clip plane offset for near clipping</param>
        /// <param name="stereoSeparationMultiplier">Stereo separation multiplier</param>
        public static void RenderReflection
        (
            ReflectionCameraInfo info,
            Transform reflectionTransform,
            Vector3 reflectionNormal,
            float clipPlaneOffset,
            float stereoSeparationMultiplier
        )
        {
            if (info.SourceCamera.stereoEnabled)
            {
                if (info.SourceCamera.stereoTargetEye == StereoTargetEyeMask.Both || info.SourceCamera.stereoTargetEye == StereoTargetEyeMask.Left)
                {
                    RenderReflectionInternal(info, reflectionTransform, reflectionNormal, clipPlaneOffset, StereoTargetEyeMask.Left, info.TargetTexture, stereoSeparationMultiplier);
                }
                if (info.SourceCamera.stereoTargetEye == StereoTargetEyeMask.Both || info.SourceCamera.stereoTargetEye == StereoTargetEyeMask.Right)
                {
                    RenderReflectionInternal(info, reflectionTransform, reflectionNormal, clipPlaneOffset, StereoTargetEyeMask.Right, info.TargetTexture2, stereoSeparationMultiplier);
                }
            }
            else
            {
                RenderReflectionInternal(info, reflectionTransform, reflectionNormal, clipPlaneOffset, StereoTargetEyeMask.Both, info.TargetTexture, stereoSeparationMultiplier);
            }
        }

        /// <summary>
        /// Detect if an xr device is connected
        /// </summary>
        /// <returns>True if xr device connected, false otherwise</returns>
        public static bool HasXRDevice()
        {

#if UNITY_2020_1_OR_NEWER

            var xrDisplaySubsystems = new List<UnityEngine.XR.XRDisplaySubsystem>();
            SubsystemManager.GetInstances<UnityEngine.XR.XRDisplaySubsystem>(xrDisplaySubsystems);
            foreach (var xrDisplay in xrDisplaySubsystems)
            {
                if (xrDisplay.running)
                {
                    return true;
                }
            }
            return false;

#else

            return UnityEngine.XR.XRDevice.isPresent;

#endif

        }

        /// <summary>
        /// Render a reflection camera. Reflection camera should already be setup with a render texture.
        /// </summary>
        /// <param name="info">Camera info</param>
        /// <param name="reflectionTransform">Reflection transform</param>
        /// <param name="reflectionNormal">Reflection normal vector</param>
        /// <param name="clipPlaneOffset">Clip plane offset for near clipping</param>
        /// <param name="eye">Stereo eye mask</param>
        /// <param name="targetTexture">Target texture</param>
        /// <param name="stereoSeparationMultiplier">Stereo separation multiplier</param>
        private static void RenderReflectionInternal
        (
            ReflectionCameraInfo info,
            Transform reflectionTransform,
            Vector3 reflectionNormal,
            float clipPlaneOffset,
            StereoTargetEyeMask eye,
            RenderTexture targetTexture,
            float stereoSeparationMultiplier
        )
        {
            bool oldInvertCulling = GL.invertCulling;

            // find out the reflection plane: position and normal in world space
            Vector3 pos = reflectionTransform.position;

            // Render reflection
            // Reflect camera around reflection plane
            if (info.SourceCameraIsReflection && GL.invertCulling)
            {
                reflectionNormal = -reflectionNormal;
            }

            float d = -Vector3.Dot(reflectionNormal, pos) - clipPlaneOffset;
            Vector4 reflectionPlane = new Vector4(reflectionNormal.x, reflectionNormal.y, reflectionNormal.z, d);

            Matrix4x4 reflection;
            CalculateReflectionMatrix(out reflection, reflectionPlane);
            Vector3 savedPos = info.SourceCamera.transform.position;
            Vector3 reflectPos = reflection.MultiplyPoint(savedPos);
            Matrix4x4 worldToCameraMatrix = info.SourceCamera.worldToCameraMatrix;
            if (eye == StereoTargetEyeMask.Left)
            {
                worldToCameraMatrix[12] += (info.SourceCamera.stereoSeparation * 0.5f * stereoSeparationMultiplier);
                info.ReflectionCamera.projectionMatrix = info.SourceCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Left);
            }
            else if (eye == StereoTargetEyeMask.Right)
            {
                worldToCameraMatrix[12] -= (info.SourceCamera.stereoSeparation * 0.5f * stereoSeparationMultiplier);
                info.ReflectionCamera.projectionMatrix = info.SourceCamera.GetStereoProjectionMatrix(Camera.StereoscopicEye.Right);
            }
            else
            {
                info.ReflectionCamera.projectionMatrix = info.SourceCamera.projectionMatrix;
            }
            info.ReflectionCamera.worldToCameraMatrix = worldToCameraMatrix * reflection;
            if (info.ReflectionCamera.actualRenderingPath != RenderingPath.DeferredShading)
            {
                // Optimization: Setup oblique projection matrix so that near plane is our reflection plane.
                // This way we clip everything below/above it for free.
                Vector4 clipPlane = CameraSpacePlane(info.ReflectionCamera, pos, reflectionNormal, clipPlaneOffset, GL.invertCulling ? -1.0f : 1.0f);
                info.ReflectionCamera.projectionMatrix = info.ReflectionCamera.CalculateObliqueMatrix(clipPlane);
            }
            for (int i = 0; i < cullDistances.Length; i++)
            {
                cullDistances[i] = info.ReflectionCamera.farClipPlane;
            }
            info.ReflectionCamera.layerCullDistances = cullDistances;
            info.ReflectionCamera.layerCullSpherical = true;

            GL.invertCulling = !GL.invertCulling;
            info.ReflectionCamera.transform.position = reflectPos;
            if (++renderCount < maxRenderCount)
            {
                info.ReflectionCamera.targetTexture = targetTexture;
                info.ReflectionCamera.Render();
                info.ReflectionCamera.targetTexture = null;
            }
            info.ReflectionCamera.transform.position = savedPos;
            GL.invertCulling = oldInvertCulling;
        }

        private void AddRenderTextureForSourceCamera(Camera sourceCamera, RenderTexture tex, StereoTargetEyeMask eyeMask)
        {
            List<KeyValuePair<RenderTexture, StereoTargetEyeMask>> tmp;
            if (!sourceCamerasToRenderTextures.TryGetValue(sourceCamera, out tmp))
            {
                sourceCamerasToRenderTextures[sourceCamera] = tmp = new List<KeyValuePair<RenderTexture, StereoTargetEyeMask>>();
            }
            tmp.Add(new KeyValuePair<RenderTexture, StereoTargetEyeMask>(tex, eyeMask));
        }

        private bool ShouldIgnoreCamera(Camera sourceCamera, out bool isReflection)
        {
            string camName;
            isReflection = CameraIsReflection(sourceCamera, out camName);

#if UNITY_EDITOR

            if (sourceCamera.cameraType == CameraType.Preview || camName.IndexOf("preview", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return true;
            }

#endif

            // ignore water and reflection cameras
            if (isReflection && (sourceCamera.transform.parent == null || sourceCamera.transform.parent.GetComponent<MagicMirrorScript>() == null))
            {
                return true;
            }

            return false;
        }

        private void CleanupCamera(ReflectionCameraInfo info, bool destroyCamera)
        {
            if (info.ReflectionCamera == null)
            {
                return;
            }
            //info.ReflectCamera.targetTexture = null;
            if (destroyCamera

#if UNITY_EDITOR

                && Application.isPlaying

#endif

            )
            {
                DestroyImmediate(info.ReflectionCamera.gameObject);
            }
        }

        private void CleanupCameras(bool destroyCameras)
        {
            cameraCache.AddRange(currentCameras);
            currentCameras.Clear();
            for (int i = cameraCache.Count - 1; i >= 0; i--)
            {
                CleanupCamera(cameraCache[i], destroyCameras);
                if (destroyCameras)
                {
                    cameraCache.RemoveAt(i);
                }
            }
        }

        private void LateUpdate()
        {

#if UNITY_EDITOR

            if (renderCount != 0)
            {
                // Debug.Log("Render count: " + renderCount);
            }

#endif

            CleanupCameras(false);
            renderCount = 0;
        }

        private void OnEnable()
        {
            if (!initialized)
            {
                initialized = true;
                ReflectRenderer = (ReflectRenderer == null ? GetComponent<Renderer>() : ReflectRenderer);

#if UNITY_EDITOR

                if (Application.isPlaying)

#endif
                { ReflectRenderer.sharedMaterial = ReflectRenderer.material; }

                ReflectRenderer.sharedMaterial.DisableKeyword(mirrorRecursionLimitKeyword);
                for (int i = 0; i < transform.childCount; i++)
                {
                    Camera cam = transform.GetChild(i).GetComponent<Camera>();
                    if (cam != null)
                    {
                        cameraCache.Add(new ReflectionCameraInfo { ReflectionCamera = cam });
                    }
                }
            }
            Camera.onPreCull += CameraPreCull;
            Camera.onPreRender += CameraPreRender;
            Camera.onPostRender += CameraPostRender;
            ReflectionSamplerName2 = (ReflectionSamplerName + "2");
        }

        private void OnDisable()
        {
            CleanupCameras(true);
            Camera.onPreCull -= CameraPreCull;
            Camera.onPreRender -= CameraPreRender;
            Camera.onPostRender -= CameraPostRender;
            sourceCamerasToRenderTextures.Clear();
        }

        private void OnDestroy()
        {

#if UNITY_EDITOR

            if (!Application.isPlaying)
            {
                return;
            }

#endif

            if (ReflectRenderer.sharedMaterial != null)
            {
                Destroy(ReflectRenderer.sharedMaterial);
            }
        }

        private void OnWillRenderObject()
        {
            QueueReflection(Camera.current);
        }

        private void CameraPreCull(Camera camera)
        {
            KeyValuePair<RenderTexture, RenderTexture> kv = new KeyValuePair<RenderTexture, RenderTexture>
            (
                ReflectRenderer.sharedMaterial.GetTexture(ReflectionSamplerName) as RenderTexture,
                ReflectRenderer.sharedMaterial.GetTexture(ReflectionSamplerName2) as RenderTexture
            );
            currentRenderTextures.Add(kv);
        }

        private void CameraPreRender(Camera camera)
        {

        }

        private void CameraPostRender(Camera camera)
        {
            if (currentRenderTextures.Count != 0)
            {
                int idx = currentRenderTextures.Count - 1;
                KeyValuePair<RenderTexture, RenderTexture> kv = currentRenderTextures[idx];
                ReflectRenderer.sharedMaterial.SetTexture(ReflectionSamplerName, kv.Key);
                ReflectRenderer.sharedMaterial.SetTexture(ReflectionSamplerName2, kv.Value);
                currentRenderTextures.RemoveAt(idx);
            }
            for (int i = currentCameras.Count - 1; i >= 0; i--)
            {
                if (currentCameras[i].SourceCamera == camera)
                {
                    CleanupCamera(currentCameras[i], false);
                    currentCameras.RemoveAt(i);
                }
            }
            List<KeyValuePair<RenderTexture, StereoTargetEyeMask>> texturesToRelease;
            if (sourceCamerasToRenderTextures.TryGetValue(camera, out texturesToRelease))
            {
                // free up temporary render textures
                // if in multi-pass, we only free the texture for the current rendering eye
                // if in single-pass, we free them all
                StereoTargetEyeMask mask = StereoTargetEyeMask.Both;
                string tmp;
                bool isReflectionCamera = CameraIsReflection(camera, out tmp);
                for (int i = texturesToRelease.Count - 1; i >= 0; i--)
                {
                    // if reflection or multi-pass, use camera eye, else free all textures
                    if (isReflectionCamera || (HasXRDevice() && UnityEngine.XR.XRSettings.eyeTextureDesc.vrUsage == VRTextureUsage.OneEye))
                    {
                        switch (camera.stereoActiveEye)
                        {
                            default:
                                mask = StereoTargetEyeMask.Both;
                                break;

                            case Camera.MonoOrStereoscopicEye.Left:
                                mask = StereoTargetEyeMask.Left;
                                break;

                            case Camera.MonoOrStereoscopicEye.Right:
                                mask = StereoTargetEyeMask.Right;
                                break;
                        }
                    }
                    KeyValuePair<RenderTexture, StereoTargetEyeMask> tex = texturesToRelease[i];
                    if (tex.Key != null && (mask & tex.Value) != StereoTargetEyeMask.None)
                    {
                        RenderTexture.ReleaseTemporary(tex.Key);
                        texturesToRelease.RemoveAt(i);
                    }
                }
            }
        }

        private void SyncCameraSettings(Camera reflectCamera, Camera sourceCamera)
        {
            reflectCamera.nearClipPlane = (NearPlane <= 0.0f ? sourceCamera.nearClipPlane : NearPlane);
            reflectCamera.farClipPlane = (FarPlane <= 0.0f ? sourceCamera.farClipPlane : FarPlane);
            reflectCamera.aspect = (AspectRatio <= 0.0f ? sourceCamera.aspect : AspectRatio);
            if (!reflectCamera.stereoEnabled)
            {
                reflectCamera.fieldOfView = (FieldOfView <= 0.0f ? sourceCamera.fieldOfView : FieldOfView);
            }
            reflectCamera.orthographic = sourceCamera.orthographic;
            reflectCamera.orthographicSize = sourceCamera.orthographicSize;
            reflectCamera.renderingPath = (ReflectionCameraRenderingPath == RenderingPath.UsePlayerSettings ? sourceCamera.renderingPath : ReflectionCameraRenderingPath);
            reflectCamera.backgroundColor = Color.red;
            reflectCamera.clearFlags = ReflectSkybox ? CameraClearFlags.Skybox : CameraClearFlags.SolidColor;
            reflectCamera.cullingMask = (CurrentRecursionLevel == 0 || ReflectionMaskRecursion.value == 0 ? ReflectionMask : ReflectionMaskRecursion);
            reflectCamera.stereoSeparation = sourceCamera.stereoSeparation;
            reflectCamera.allowHDR = sourceCamera.allowHDR;
            reflectCamera.allowMSAA = (AntiAliasingSamples > AntiAliasingSampleCount.None);
            reflectCamera.rect = new Rect(0.0f, 0.0f, 1.0f, 1.0f);
            reflectCamera.transform.rotation = sourceCamera.transform.rotation;
            reflectCamera.transform.position = sourceCamera.transform.position;

            if (ReflectSkybox)
            {
                if (sourceCamera.gameObject.GetComponent(typeof(Skybox)))
                {
                    Skybox sb = (Skybox)reflectCamera.gameObject.GetComponent(typeof(Skybox));
                    if (!sb)
                    {
                        sb = (Skybox)reflectCamera.gameObject.AddComponent(typeof(Skybox));
                        sb.hideFlags = HideFlags.HideAndDontSave;
                    }
                    sb.material = ((Skybox)sourceCamera.GetComponent(typeof(Skybox))).material;
                }
            }
        }

        private ReflectionCameraInfo CreateReflectionCamera(Camera sourceCamera, bool sourceCameraIsReflection)
        {
            // don't render if we are not enabled
            if (ReflectRenderer == null || !ReflectRenderer.enabled || ReflectRenderer.sharedMaterial == null || sourceCamera == null)
            {
                return null;
            }

            // only render reflection cameras with this script
            MagicMirrorScript reflScript = (sourceCameraIsReflection && sourceCamera.transform.parent != null ? sourceCamera.transform.parent.GetComponent<MagicMirrorScript>() : null);
            if (sourceCameraIsReflection && (reflScript == null || reflScript == this))
            {
                // don't render ourselves in our camera
                ReflectRenderer.sharedMaterial.EnableKeyword(mirrorRecursionLimitKeyword);
                return null;
            }

            // recursion limit hit, bail...
            if (reflScript != null && currentCameras.Count >=

#if UNITY_EDITOR

                (Application.isPlaying ? RecursionLimit : 0)

#else

                RecursionLimit

#endif

            )
            {
                ReflectRenderer.sharedMaterial.EnableKeyword(mirrorRecursionLimitKeyword);
                return null;
            }

            ReflectionCameraInfo info;
            if (cameraCache.Count == 0)
            {
                GameObject obj = new GameObject("MirrorReflectionCamera");
                obj.hideFlags = HideFlags.HideAndDontSave;
                obj.SetActive(false);
                obj.transform.parent = transform;
                Camera newReflectionCamera = obj.AddComponent<Camera>();
                newReflectionCamera.enabled = false;
                info = new ReflectionCameraInfo
                {
                    SourceCamera = sourceCamera,
                    ReflectionCamera = newReflectionCamera
                };
            }
            else
            {
                int idx = cameraCache.Count - 1;
                info = cameraCache[idx];
                cameraCache.RemoveAt(idx);
                CleanupCamera(info, false);
            }
            info.SourceCamera = sourceCamera;
            info.SourceCameraIsReflection = sourceCameraIsReflection;
            int size = Math.Max(32, (int)(Mathf.Pow(RecursionRenderTextureSizeReducerPower, (float)CurrentRecursionLevel) * (float)RenderTextureSize));
            info.TargetTexture = RenderTexture.GetTemporary(size, size, 16, RenderTextureFormat.DefaultHDR);
            info.TargetTexture.wrapMode = TextureWrapMode.Clamp;
            info.TargetTexture.filterMode = FilterMode.Bilinear;
            if (AntiAliasingSamples > AntiAliasingSampleCount.None)
            {
                info.TargetTexture.antiAliasing = (int)AntiAliasingSamples;
            }
            AddRenderTextureForSourceCamera(sourceCamera, info.TargetTexture, StereoTargetEyeMask.Left);
            if (sourceCamera.stereoEnabled)
            {
                info.TargetTexture2 = RenderTexture.GetTemporary(size, size, 16, RenderTextureFormat.DefaultHDR);
                info.TargetTexture2.wrapMode = TextureWrapMode.Clamp;
                info.TargetTexture2.filterMode = FilterMode.Bilinear;
                if (AntiAliasingSamples > AntiAliasingSampleCount.None)
                {
                    info.TargetTexture2.antiAliasing = (int)AntiAliasingSamples;
                }
                AddRenderTextureForSourceCamera(sourceCamera, info.TargetTexture2, StereoTargetEyeMask.Right);
            }
            else
            {
                info.TargetTexture2 = info.TargetTexture;
            }
            currentCameras.Add(info);
            return info;
        }

        private void RenderReflectionCamera(ReflectionCameraInfo info)
        {
            // bail if we don't have a camera or renderer
            if (info == null || info.ReflectionCamera == null || info.SourceCamera == null || ReflectRenderer == null || ReflectRenderer.sharedMaterial == null || !ReflectRenderer.enabled)
            {
                return;
            }

            CurrentRecursionLevel = currentCameras.Count + (info.SourceCameraIsReflection ? 1 : 0);
            renderCount++;
            Camera sourceCamera = info.SourceCamera;
            Camera reflectionCamera = info.ReflectionCamera;
            int oldPixelLightCount = QualitySettings.pixelLightCount;
            int oldCullingMask = reflectionCamera.cullingMask;
            bool oldSoftParticles = QualitySettings.softParticles;
            int oldAntiAliasing = QualitySettings.antiAliasing;
            ShadowQuality oldShadows = QualitySettings.shadows;
            SyncCameraSettings(reflectionCamera, sourceCamera);

            // MAGIC MIRROR RECURSION OPTIMIZATION
            if (currentCameras.Count > 1)
            {
                if (currentCameras.Count > 3)
                {
                    QualitySettings.shadows = ShadowQuality.Disable;
                }
                QualitySettings.shadows = ShadowQuality.HardOnly;
                QualitySettings.antiAliasing = 0;
                QualitySettings.softParticles = false;
                QualitySettings.pixelLightCount = 0;
                reflectionCamera.cullingMask &= waterLayerInverse;
            }
            else
            {
                QualitySettings.pixelLightCount = MaximumPerPixelLightsToReflect;
            }
            Transform reflectionTransform = transform;
            Vector3 reflectionNormal = (NormalIsForward ? -reflectionTransform.forward : reflectionTransform.up);

            // use shared reflection render function
            RenderReflection(info, reflectionTransform, reflectionNormal, ClipPlaneOffset, StereoSeparationMultiplier);

            // restore render state
            reflectionCamera.cullingMask = oldCullingMask;
            QualitySettings.pixelLightCount = oldPixelLightCount;
            QualitySettings.softParticles = oldSoftParticles;
            QualitySettings.antiAliasing = oldAntiAliasing;
            QualitySettings.shadows = oldShadows;
            ReflectRenderer.sharedMaterial.SetTexture(ReflectionSamplerName, info.TargetTexture);
            ReflectRenderer.sharedMaterial.SetTexture(ReflectionSamplerName2, info.TargetTexture2);
            ReflectRenderer.sharedMaterial.DisableKeyword(mirrorRecursionLimitKeyword);

            currentCameras.Remove(info);
            info.SourceCamera = null;
            info.TargetTexture = null;
            info.TargetTexture2 = null;
            cameraCache.Add(info);
        }

        // Given position/normal of the plane, calculates plane in camera space.
        private static Vector4 CameraSpacePlane(Camera cam, Vector3 pos, Vector3 normal, float clipPlaneOffset, float sideSign)
        {
            Vector3 offsetPos = pos + normal * clipPlaneOffset;
            Matrix4x4 m = cam.worldToCameraMatrix;
            Vector3 cpos = m.MultiplyPoint(offsetPos);
            Vector3 cnormal = m.MultiplyVector(normal).normalized * sideSign;
            return new Vector4(cnormal.x, cnormal.y, cnormal.z, -Vector3.Dot(cpos, cnormal));
        }

        // Calculates reflection matrix around the given plane
        private static void CalculateReflectionMatrix(out Matrix4x4 reflectionMat, Vector4 plane)
        {
            reflectionMat.m00 = (1F - 2F * plane[0] * plane[0]);
            reflectionMat.m01 = (-2F * plane[0] * plane[1]);
            reflectionMat.m02 = (-2F * plane[0] * plane[2]);
            reflectionMat.m03 = (-2F * plane[3] * plane[0]);

            reflectionMat.m10 = (-2F * plane[1] * plane[0]);
            reflectionMat.m11 = (1F - 2F * plane[1] * plane[1]);
            reflectionMat.m12 = (-2F * plane[1] * plane[2]);
            reflectionMat.m13 = (-2F * plane[3] * plane[1]);

            reflectionMat.m20 = (-2F * plane[2] * plane[0]);
            reflectionMat.m21 = (-2F * plane[2] * plane[1]);
            reflectionMat.m22 = (1F - 2F * plane[2] * plane[2]);
            reflectionMat.m23 = (-2F * plane[3] * plane[2]);

            reflectionMat.m30 = 0F;
            reflectionMat.m31 = 0F;
            reflectionMat.m32 = 0F;
            reflectionMat.m33 = 1F;
        }
    }
}

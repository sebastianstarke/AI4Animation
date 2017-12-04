using UnityEngine.Rendering;

namespace UnityEngine.PostProcessing
{
    using DebugMode = BuiltinDebugViewsModel.Mode;

    public sealed class DepthOfFieldComponent : PostProcessingComponentRenderTexture<DepthOfFieldModel>
    {
        static class Uniforms
        {
            internal static readonly int _DepthOfFieldTex = Shader.PropertyToID("_DepthOfFieldTex");
            internal static readonly int _Distance = Shader.PropertyToID("_Distance");
            internal static readonly int _LensCoeff = Shader.PropertyToID("_LensCoeff");
            internal static readonly int _MaxCoC = Shader.PropertyToID("_MaxCoC");
            internal static readonly int _RcpMaxCoC = Shader.PropertyToID("_RcpMaxCoC");
            internal static readonly int _RcpAspect = Shader.PropertyToID("_RcpAspect");
            internal static readonly int _MainTex = Shader.PropertyToID("_MainTex");
            internal static readonly int _HistoryCoC = Shader.PropertyToID("_HistoryCoC");
        }

        const string k_ShaderString = "Hidden/Post FX/Depth Of Field";

        public override bool active
        {
            get
            {
                return model.enabled
                       && SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.ARGBHalf)
                       && SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RHalf)
                       && !context.interrupted;
            }
        }

        public override DepthTextureMode GetCameraFlags()
        {
            return DepthTextureMode.Depth;
        }

        RenderTexture m_CoCHistory;
        RenderBuffer[] m_MRT = new RenderBuffer[2];

        // Height of the 35mm full-frame format (36mm x 24mm)
        const float k_FilmHeight = 0.024f;

        float CalculateFocalLength()
        {
            var settings = model.settings;

            if (!settings.useCameraFov)
                return settings.focalLength / 1000f;

            float fov = context.camera.fieldOfView * Mathf.Deg2Rad;
            return 0.5f * k_FilmHeight / Mathf.Tan(0.5f * fov);
        }

        float CalculateMaxCoCRadius(int screenHeight)
        {
            // Estimate the allowable maximum radius of CoC from the kernel
            // size (the equation below was empirically derived).
            float radiusInPixels = (float)model.settings.kernelSize * 4f + 10f;

            // Applying a 5% limit to the CoC radius to keep the size of
            // TileMax/NeighborMax small enough.
            return Mathf.Min(0.05f, radiusInPixels / screenHeight);
        }

        public void Prepare(RenderTexture source, Material uberMaterial, bool antialiasCoC)
        {
            var settings = model.settings;

            // Material setup
            var material = context.materialFactory.Get(k_ShaderString);
            material.shaderKeywords = null;

            var s1 = settings.focusDistance;
            var f = CalculateFocalLength();
            s1 = Mathf.Max(s1, f);
            material.SetFloat(Uniforms._Distance, s1);

            var coeff = f * f / (settings.aperture * (s1 - f) * k_FilmHeight * 2);
            material.SetFloat(Uniforms._LensCoeff, coeff);

            var maxCoC = CalculateMaxCoCRadius(source.height);
            material.SetFloat(Uniforms._MaxCoC, maxCoC);
            material.SetFloat(Uniforms._RcpMaxCoC, 1f / maxCoC);

            var rcpAspect = (float)source.height / source.width;
            material.SetFloat(Uniforms._RcpAspect, rcpAspect);

            var rt1 = context.renderTextureFactory.Get(context.width / 2, context.height / 2, 0, RenderTextureFormat.ARGBHalf);
            source.filterMode = FilterMode.Point;

            // Pass #1 - Downsampling, prefiltering and CoC calculation
            Graphics.Blit(source, rt1, material, 0);

            // Pass #2 - CoC Antialiasing
            var pass = rt1;
            if (antialiasCoC)
            {
                pass = context.renderTextureFactory.Get(context.width / 2, context.height / 2, 0, RenderTextureFormat.ARGBHalf);

                if (m_CoCHistory == null || !m_CoCHistory.IsCreated() || m_CoCHistory.width != context.width / 2 || m_CoCHistory.height != context.height / 2)
                {
                    m_CoCHistory = RenderTexture.GetTemporary(context.width / 2, context.height / 2, 0, RenderTextureFormat.RHalf);
                    m_CoCHistory.filterMode = FilterMode.Point;
                    m_CoCHistory.name = "CoC History";
                    Graphics.Blit(rt1, m_CoCHistory, material, 6);
                }

                var tempCoCHistory = RenderTexture.GetTemporary(context.width / 2, context.height / 2, 0, RenderTextureFormat.RHalf);
                tempCoCHistory.filterMode = FilterMode.Point;
                tempCoCHistory.name = "CoC History";

                m_MRT[0] = pass.colorBuffer;
                m_MRT[1] = tempCoCHistory.colorBuffer;
                material.SetTexture(Uniforms._MainTex, rt1);
                material.SetTexture(Uniforms._HistoryCoC, m_CoCHistory);
                Graphics.SetRenderTarget(m_MRT, rt1.depthBuffer);
                GraphicsUtils.Blit(material, 5);

                RenderTexture.ReleaseTemporary(m_CoCHistory);
                m_CoCHistory = tempCoCHistory;
            }

            // Pass #3 - Bokeh simulation
            var rt2 = context.renderTextureFactory.Get(context.width / 2, context.height / 2, 0, RenderTextureFormat.ARGBHalf);
            Graphics.Blit(pass, rt2, material, 1 + (int)settings.kernelSize);

            if (context.profile.debugViews.IsModeActive(DebugMode.FocusPlane))
            {
                uberMaterial.SetTexture(Uniforms._DepthOfFieldTex, rt1);
                uberMaterial.SetFloat(Uniforms._MaxCoC, maxCoC);
                uberMaterial.EnableKeyword("DEPTH_OF_FIELD_COC_VIEW");
                context.Interrupt();
            }
            else
            {
                uberMaterial.SetTexture(Uniforms._DepthOfFieldTex, rt2);
                uberMaterial.EnableKeyword("DEPTH_OF_FIELD");
            }

            if (antialiasCoC)
                context.renderTextureFactory.Release(pass);

            context.renderTextureFactory.Release(rt1);
            source.filterMode = FilterMode.Bilinear;
        }

        public override void OnDisable()
        {
            if (m_CoCHistory != null)
                RenderTexture.ReleaseTemporary(m_CoCHistory);

            m_CoCHistory = null;
        }
    }
}

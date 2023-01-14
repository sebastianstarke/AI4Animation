using UnityEngine;

public class Transparency : MonoBehaviour {
	
	public enum RenderingMode{Opaque, Cutout, Fade, Transparent};

	[RangeAttribute(0f,1f)] public float Alpha = 1f;

	void Awake() {
		ApplyRecursively(transform);
	}

	void OnValidate() {
		if(!Application.isPlaying) {
			return;
		}
		SetTransparency(Alpha);
	}

	public void SetTransparency(float alpha) {
        if(Application.isPlaying) {
            Alpha = alpha;
            ApplyRecursively(transform);
        }
	}

	private void ApplyRecursively(Transform t) {
		Renderer renderer = t.GetComponent<Renderer>();
		if(renderer != null) {
            foreach(Material material in renderer.materials) {
                if(material.HasProperty("_Color")) {
                    Color color = material.color;
                    material.SetColor("_Color", new Vector4(color.r, color.g, color.b, Alpha));
                    if(Alpha == 1f) {
                        ChangeRenderMode(material, RenderingMode.Opaque);
                    } else {
                        ChangeRenderMode(material, RenderingMode.Transparent);
                    }
                }
            }
		}
		for(int i=0; i<t.childCount; i++) {
			ApplyRecursively(t.GetChild(i));
		}
	}

    private void ChangeRenderMode(Material material, RenderingMode mode)
     {
		 material.SetFloat("_Mode", (float)mode);
         switch (mode)
         {
             case RenderingMode.Opaque:
                 material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                 material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
                 material.SetInt("_ZWrite", 1);
                 material.DisableKeyword("_ALPHATEST_ON");
                 material.DisableKeyword("_ALPHABLEND_ON");
                 material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                 material.renderQueue = -1;
                 break;
             case RenderingMode.Cutout:
                 material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                 material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.Zero);
                 material.SetInt("_ZWrite", 1);
                 material.EnableKeyword("_ALPHATEST_ON");
                 material.DisableKeyword("_ALPHABLEND_ON");
                 material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                 material.renderQueue = 2450;
                 break;
             case RenderingMode.Fade:
                 material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                 material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                 material.SetInt("_ZWrite", 0);
                 material.DisableKeyword("_ALPHATEST_ON");
                 material.EnableKeyword("_ALPHABLEND_ON");
                 material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
                 material.renderQueue = 3000;
                 break;
             case RenderingMode.Transparent:
                 material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                 material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                 material.SetInt("_ZWrite", 0);
                 material.DisableKeyword("_ALPHATEST_ON");
                 material.DisableKeyword("_ALPHABLEND_ON");
                 material.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                 material.renderQueue = 3000;
                 break;
         }
	 }

}

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

using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(OVROverlayCanvas))]
public class OVROverlayCanvasEditor : Editor {

	GUIStyle mWarningBoxStyle;

	void OnEnable()
	{
		var warningBoxStyleTex = new Texture2D(1, 1);
		warningBoxStyleTex.SetPixel(0, 0, new Color(0.4f, 0.4f, 0.4f, 0.2f));
		warningBoxStyleTex.Apply();
		mWarningBoxStyle = new GUIStyle();
		mWarningBoxStyle.normal.background = warningBoxStyleTex;
		mWarningBoxStyle.padding = new RectOffset(8, 8, 2, 2);
		mWarningBoxStyle.margin = new RectOffset(4, 4, 4, 4);
	}

	public override void OnInspectorGUI()
	{
		OVROverlayCanvas canvas = target as OVROverlayCanvas;

		EditorGUI.BeginChangeCheck();

		float lastTextureSize = canvas.MaxTextureSize;
		canvas.MaxTextureSize = EditorGUILayout.IntField(new GUIContent("Max Texture Size", "Limits the maximum size of the texture used for this canvas"), canvas.MaxTextureSize);
		canvas.MinTextureSize = EditorGUILayout.IntField(new GUIContent("Min Texture Size", "Limits the minimum size this texture will be displayed at"), canvas.MinTextureSize);

		// Automatically adjust pixels per unit when texture size is adjusted to maintain the same density
		canvas.PixelsPerUnit *= lastTextureSize / (float)canvas.MaxTextureSize;
		canvas.PixelsPerUnit = EditorGUILayout.FloatField(new GUIContent("Pixels Per Unit", "Controls the density of the texture"), canvas.PixelsPerUnit);

		canvas.DrawRate = EditorGUILayout.IntField(new GUIContent("Draw Rate", "How often we should re-render this canvas to a texture. The canvas' transform can be changed every frame, regardless of Draw Rate. A value of 1 means every frame, 2 means every other, etc."), canvas.DrawRate);
		if (canvas.DrawRate > 1)
		{
			canvas.DrawFrameOffset = EditorGUILayout.IntField(new GUIContent("Draw Frame Offset", "Allows you to alternate which frame each canvas will draw on by specifying a frame offset."), canvas.DrawFrameOffset);
		}

		canvas.Expensive = EditorGUILayout.Toggle(new GUIContent("Expensive", "Improve the visual appearance at the cost of additional GPU time"), canvas.Expensive);
		canvas.Opacity = (OVROverlayCanvas.DrawMode)EditorGUILayout.EnumPopup(new GUIContent("Opacity", "Treat this canvas as opaque, which is a big performance improvement"), canvas.Opacity);

		if (canvas.Opacity == OVROverlayCanvas.DrawMode.TransparentDefaultAlpha)
		{
			DisplayMessage(eMessageType.Notice, "Transparent Default Alpha is not recommended with overlapping semitransparent graphics.");
		}

		if (canvas.Opacity == OVROverlayCanvas.DrawMode.TransparentCorrectAlpha)
		{
			var graphics = canvas.GetComponentsInChildren<UnityEngine.UI.Graphic>();
			bool usingDefaultMaterial = false;
			foreach(var graphic in graphics)
			{
				if (graphic.material == null || graphic.material == graphic.defaultMaterial)
				{
					usingDefaultMaterial = true;
					break;
				}
			}

			if (usingDefaultMaterial)
			{
				DisplayMessage(eMessageType.Warning, "Some graphics in this canvas are using the default UI material.\nWould you like to replace all of them with the corrected UI Material?");

				if (GUILayout.Button("Replace Materials"))
				{
					var matList = AssetDatabase.FindAssets("t:Material UI Default Correct");
					if (matList.Length > 0)
					{
						var mat = AssetDatabase.LoadAssetAtPath<Material>(AssetDatabase.GUIDToAssetPath(matList[0]));

						foreach(var graphic in graphics)
						{
							if (graphic.material == null || graphic.material == graphic.defaultMaterial)
							{
								graphic.material = mat;
							}
						}
					}
				}
			}
		}
		if (canvas.Opacity == OVROverlayCanvas.DrawMode.TransparentCorrectAlpha ||
			canvas.Opacity == OVROverlayCanvas.DrawMode.TransparentDefaultAlpha)
		{
			if (PlayerSettings.colorSpace == ColorSpace.Gamma)
			{
				DisplayMessage(eMessageType.Warning, "This project's ColorSpace is set to Gamma. Oculus recommends using Linear ColorSpace. Alpha blending will not be correct in Gamma ColorSpace.");
			}
		}


		canvas.Layer = EditorGUILayout.LayerField(new GUIContent("Overlay Layer", "The layer this overlay should be drawn on"), canvas.Layer);

		if (canvas.Layer == canvas.gameObject.layer)
		{
			DisplayMessage(eMessageType.Error, $"This GameObject's Layer is the same as Overlay Layer ('{LayerMask.LayerToName(canvas.Layer)}'). "
				+ "To control camera visibility, this GameObject should have a Layer that is not the Overlay Layer.");
		}

		if (Camera.main != null)
		{
			if ((Camera.main.cullingMask & (1 << canvas.gameObject.layer)) != 0)
			{
				DisplayMessage(eMessageType.Warning, 
					$"Main Camera '{Camera.main.name}' does not cull this GameObject's Layer '{LayerMask.LayerToName(canvas.gameObject.layer)}'. "
					+ "This Canvas might be rendered by both the Main Camera and the OVROverlay system.");
			}
			if ((Camera.main.cullingMask & (1 << canvas.Layer)) == 0)
			{
				DisplayMessage(eMessageType.Error, $"Overlay Layer '{LayerMask.LayerToName(canvas.Layer)}' is culled by your main camera. "
					+ "The Overlay Layer is expected to render in the scene, so it shouldn't be culled.");
			}
		}
		else
		{
			DisplayMessage(eMessageType.Warning, "No Main Camera found. Make sure your camera does not draw this GameObject's Layer ("
				+LayerMask.LayerToName(canvas.gameObject.layer) + "), or this canvas might be rendered twice.");
		}

		if (Application.isPlaying)
		{
			EditorGUILayout.Space();
			EditorGUILayout.LabelField("Editor Debug", EditorStyles.boldLabel);
			canvas.overlayEnabled = EditorGUILayout.Toggle("Overlay Enabled", canvas.overlayEnabled);			
		}
	}

	public enum eMessageType
	{
		Notice,
		Warning,
		Error
	}

	void DisplayMessage(eMessageType messageType, string messageText)
	{
		string iconUri = "";
		string header = "";
		switch (messageType)
		{
			case eMessageType.Error:
				iconUri = "console.erroricon.sml";
				header = "Error";
				break;
			case eMessageType.Warning:
				iconUri = "console.warnicon.sml";
				header = "Warning";
				break;
			case eMessageType.Notice:
			default:
				iconUri = "console.infoicon.sml";
				header = "Notice";
				break;
		}

		EditorGUILayout.BeginHorizontal(mWarningBoxStyle); //2-column wrapper
		EditorGUILayout.BeginVertical(GUILayout.Width(20)); //column 1: icon
		EditorGUILayout.LabelField(EditorGUIUtility.IconContent(iconUri), GUILayout.Width(20));
		EditorGUILayout.EndVertical(); //end column 1: icon
		EditorGUILayout.BeginVertical(GUILayout.ExpandWidth(true)); //column 2: label, message
		GUILayout.Label(header, EditorStyles.boldLabel);
		GUILayout.Label(messageText, EditorStyles.wordWrappedLabel);
		EditorGUILayout.EndVertical(); //end column 2: label, message, objects
		EditorGUILayout.EndHorizontal(); //end 2-column wrapper
	}
}

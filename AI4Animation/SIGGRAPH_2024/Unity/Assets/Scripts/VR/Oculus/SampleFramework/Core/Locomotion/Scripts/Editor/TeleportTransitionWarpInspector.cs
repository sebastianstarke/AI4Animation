/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;
using System.Linq;
using UnityEditor;

[CustomEditor(typeof(TeleportTransitionWarp))]
public class TeleportTransitionWarpInspector : Editor
{
	public override void OnInspectorGUI()
	{
		base.OnInspectorGUI();

		var warp = (TeleportTransitionWarp) target;

		warp.PositionLerp = EditorGUILayout.CurveField("Position Lerp", warp.PositionLerp, GUILayout.Height(50));

		GUILayout.BeginHorizontal();
		GUILayout.Label("Position Lerp Modes");
		if (GUILayout.Button("Default"))
		{
			warp.PositionLerp = AnimationCurve.Linear(0, 0, 1, 1);
		}
		if (GUILayout.Button("Ease"))
		{
			warp.PositionLerp = AnimationCurve.EaseInOut(0, 0, 1, 1);
		}
		if (GUILayout.Button("Step 5"))
		{
			CreateStep(warp, 5);
		}
		if (GUILayout.Button("Step 10"))
		{
			CreateStep(warp, 10);
		}
		GUILayout.EndHorizontal();
	}

	void CreateStep(TeleportTransitionWarp warp, int count)
	{
		Keyframe[] keys = new Keyframe[count+1];
		for (int i = 0; i < count; i++)
		{
			keys[i] = new Keyframe((float)i / (count), (i + 1.0f) / count);
		}
		keys[count] = new Keyframe(1,1);
		warp.PositionLerp = new AnimationCurve(keys);
		for (int i = 0; i < count+1; i++)
		{
			AnimationUtility.SetKeyLeftTangentMode(warp.PositionLerp, i, AnimationUtility.TangentMode.Constant);
		}
	}
}

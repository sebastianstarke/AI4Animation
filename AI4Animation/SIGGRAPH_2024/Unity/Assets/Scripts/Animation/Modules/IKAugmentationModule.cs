using UnityEngine;
using MagicIK;
using System.Linq;
using System.Collections.Generic;
using UnityEngine.InputSystem.XR;


#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
using UnityEngine.SceneManagement;

namespace AI4Animation {
	public class IKAugmentationModule : Module {
		public LayerMask ObjectMask = 0;
		public InteractableObject[] InteractableObjects = new InteractableObject[0];
		public bool Smooth = true;
		
        public ContactModule GetContactModule() {
			ContactModule module = Asset.GetModule<ContactModule>();
            if(module == null) {
                Debug.Log("Contact Module not found!");
            }
            return module;
        }

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return null;
		}

#if UNITY_EDITOR
		protected override void DerivedInitialize() {
            
		}

		protected override void DerivedLoad(MotionEditor editor) {

		}
		
		protected override void DerivedUnload(MotionEditor editor) {

		}
		protected override void DerivedCallback(MotionEditor editor) {
			if(GetContactModule() == null)
				return;

			SyncObjects();
			foreach (InteractableObject obj in InteractableObjects)
			{
				obj.SetTransform(editor.GetTimestamp(), editor.Mirror);
			}

			GenericIK[] IKComponents = editor.GetSession().GetActor().gameObject.GetComponents<GenericIK>();
			
			foreach (GenericIK IK in IKComponents)
			{
				TimeSeries series = Smooth ? editor.GetTimeSeries() : null;
				string[] objectiveNames = IK.Solver.Objectives.Select(i => i.Node.Transform.name).ToArray();
				string[] contacts = ArrayExtensions.GatherByLength(objectiveNames, 1, objectiveNames.Length - 1);
				Matrix4x4[] targets = contacts.Select(x => ComputeTarget(editor.GetTimestamp(), editor.Mirror, x, x, series)).ToArray();
				
				//for wrist
				string wrist = objectiveNames[0];
				ArrayExtensions.Insert(ref targets, GetAverageTargetTransformation(editor.GetTimestamp(), editor.Mirror, contacts, wrist, series), 0);

				for (int i = 0; i < IK.Solver.Objectives.Length; i++)
				{
					IK.Solver.Objectives[i].Position = targets[i].GetPosition();
					IK.Solver.Objectives[i].Rotation = targets[i].GetRotation();
				}
				IK.Solve();
			}
		}

		private Matrix4x4 ComputeTarget(float timestamp, bool mirror, string contact, string bone, TimeSeries smoothing) {
			if(smoothing == null)  {
				return GetTargetTransformation(timestamp, mirror, contact, bone);
			}
			Matrix4x4[] trajectory = new Matrix4x4[smoothing.KeyCount];

			for(int i=0; i<trajectory.Length; i++) {
				trajectory[i] = GetTargetTransformation(timestamp + smoothing.GetKey(i).Timestamp, mirror, contact, bone);				
			}
			
			//float[] weights = new float[smoothing.KeyCount];
			float t = ContactChangeRatio(timestamp, mirror, contact, smoothing);
			// float[] gaussValues = new float[smoothing.KeyCount];
			// for (int i = 0; i < gaussValues.Length; i++)
			// {
			// 	gaussValues[i] = i.Ratio(0, gaussValues.Length - 1).Normalize(0f, 1f, -2f, 2f);
			// 	gaussValues[i] = Mathf.Pow(Mathf.Exp(-Mathf.Pow(gaussValues[i], 2f)), 100f);
			// }

			// for (int i = 0; i < weights.Length; i++)
			// {
			// 	weights[i] = Mathf.Lerp(gaussValues[i], 1f, t);
			// }
	
			// Weights = weights;
			// return trajectory.Mean(weights);
			return Utility.Interpolate(GetTargetTransformation(timestamp, mirror, contact, bone), trajectory.Mean(), t);
		}
		// f(x) = (1-t) + t (e ^-x^2)^100
		// high value => averaging
		// low value => steepness
		public float ContactChangeRatio(float timestamp, bool mirror, string contact, TimeSeries smoothing)  {
			float[] values = ContactDistribution(timestamp, mirror, contact, smoothing);
			float oneRatio = values.Ratio(1f);
			float zeroRatio = values.Ratio(0f);
			return (zeroRatio * oneRatio).Normalize(0, 0.25f, 0f, 1f);
		}

		public float ContactActivation(float timestamp, bool mirror, string contact, TimeSeries smoothing) {
			if(smoothing == null) {
				return ContactDistribution(timestamp, mirror, contact, smoothing).Sum();
			}
			return ContactDistribution(timestamp, mirror, contact, smoothing).Sum().Normalize(0f, smoothing.KeyCount, 0f, 1f);
		}

		public float[] ContactDistribution(float timestamp, bool mirror, string contact, TimeSeries smoothing) {
			if(smoothing == null)  {
				return new float[]{GetContactModule().GetContact(timestamp, mirror, contact)};
			}
			float[] values = new float[smoothing.KeyCount];
			for(int i=0; i<values.Length; i++) {
				values[i] = GetContactModule().GetContact(timestamp + smoothing.GetKey(i).Timestamp, mirror, contact);
			}
			return values;
		}

		//replace with loop over compute target
		private Matrix4x4 GetAverageTargetTransformation(float timestamp, bool mirror, string[] contacts, string bone, TimeSeries smoothing) {
			Matrix4x4[] transformations = new Matrix4x4[contacts.Length];
			for (int i = 0; i < contacts.Length; i++)
			{
				transformations[i] = ComputeTarget(timestamp, mirror, contacts[i], bone, smoothing);
			}
			return transformations.Mean();
		}

		// transition => most averaging
		private Matrix4x4 GetTargetTransformation(float timestamp, bool mirror, string contact, string bone = null) {
			if(bone == null){
				bone = contact;
			}
			string[] candidates = GetContactModule().GetContactNames(timestamp, mirror, contact);
			List<Matrix4x4> transformations = new List<Matrix4x4>();
			for (int i = 0; i < candidates.Length; i++)
			{
				InteractableObject obj = GetObject(candidates[i]);
				if(obj != null) {
					transformations.Add(GetBoneTransformation(timestamp, mirror, Asset.Source.GetBoneIndex(bone), obj));
				}
			}
			//return average of contacts
			if(transformations.Count > 0) {
				return transformations.ToArray().Mean();
			}
			//return gt
			return Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirror);
		}

		// private string[] GetObjectNamesWithMostContacts(float timestamp, bool mirror, string[] contacts){
		// 	string[][] contactNames = GetContactModule().GetContactNames(timestamp, mirror, contacts);
		// 	//flat array
		// 	string[] names = Flatten(contactNames);
		// 	//sort by value and count
		// 	var dict = names.GroupBy(x => x)
		// 	.Select(g => new {Value = g.Key, Count = g.Count()})
		// 	.OrderByDescending(x=>x.Count);

		// 	//get items with highest count
		// 	var max = dict.Where(x => x.Count == dict.Select(y => y.Count).ToArray().Max()).ToArray();
		// 	return max.Select(x => x.Value).ToArray();
			
		// 	string[] Flatten(string[][] array) {
		// 		List<string> result = new List<string>();
		// 		for (int i = 0; i < array.Length; i++)
		// 		{
		// 			for (int j = 0; j < array[i].Length; j++)
		// 			{	
		// 				string element = array[i][j];
		// 				if(element != null && element != "")
		// 					result.Add(element);
		// 			}
		// 		}
		// 		return result.ToArray();
		// 	}
		// }

		protected override void DerivedGUI(MotionEditor editor) {

		}

		protected override void DerivedDraw(MotionEditor editor) {
			GenericIK[] IKComponents = editor.GetSession().GetActor().gameObject.GetComponents<GenericIK>();
			foreach (GenericIK IK in IKComponents)
			{
				IK.Solver.Draw();

				string[] bones = IK.Solver.Objectives.Select(i => i.Node.Transform.name).ToArray();
				editor.GetSession().GetActor().Draw(
					Asset.GetFrame(editor.GetTimestamp()).GetBoneTransformations(bones, editor.Mirror),
					bones,
					UltiDraw.Green,
					UltiDraw.White,
					Actor.DRAW.Skeleton
				);
			}
		}

		protected override void DerivedInspector(MotionEditor editor) {
			Undo.RecordObject(this, this.name);
			Smooth = EditorGUILayout.Toggle("Smooth", Smooth);
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.HelpBox("Interactable Objects: ", MessageType.None);
			ObjectMask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField(InternalEditorUtility.LayerMaskToConcatenatedLayersMask(ObjectMask), InternalEditorUtility.layers));
			EditorGUILayout.EndHorizontal();
			for(int i=0; i<InteractableObjects.Length; i++) {
				InteractableObjects[i].Inspector(editor);
			}
		}
#endif

		public Matrix4x4[] GetBoneTransformations(float timestamp, bool mirror, int[] boneIndices, InteractableObject obj) {
			return boneIndices.Select(i => GetBoneTransformation(timestamp, mirror, i, obj)).ToArray();
		}
		
		public Matrix4x4 GetBoneTransformation(float timestamp, bool mirror, int boneIdx, InteractableObject obj) {
			return Asset.GetFrame(timestamp).GetBoneTransformation(boneIdx, mirror).TransformationTo(
				obj.GetSourceTransformation(timestamp, mirror)).TransformationFrom(
					obj.GetTransformation(timestamp, mirror)
				);
		}

		public InteractableObject AddObject(string joint) {
			return AddObject(joint, Vector3.zero, Vector3.zero);
		}

		public InteractableObject AddObject(string joint, Vector3 position, Vector3 rotation) {
			MotionAsset.Hierarchy.Bone bone =  Asset.Source.FindBone(joint);
			if(bone == null) {
				Debug.Log("Bone " + joint + " cannot be found in Asset Hierachy.");
				return null;
			} else {
				InteractableObject obj = new InteractableObject(this, bone.Index, position, rotation);
				ArrayExtensions.Append(ref InteractableObjects, obj);
				return obj;
			}
		}

		public void RemoveObject(InteractableObject obj) {
			if(!ArrayExtensions.Remove(ref InteractableObjects, obj)) {
				Debug.Log("Object could not be found in " + Asset.name + ".");
			}
		}

		public InteractableObject GetObject(string bone) {
			return System.Array.Find(InteractableObjects, x => x.GetName() == bone);
		}

		public InteractableObject[] GetObjects(params string[] bones) {
			InteractableObject[] objects = new InteractableObject[bones.Length];
			for(int i=0; i<objects.Length; i++) {
				objects[i] = GetObject(bones[i]);
			}
			return objects;
		}

		public string[] GetObjectNames() {
			string[] names = new string[InteractableObjects.Length];
			for(int i=0; i<InteractableObjects.Length; i++) {
				names[i] = InteractableObjects[i].GetName();
			}
			return names;
		}

		public GameObject GetSceneObject(InteractableObject obj) {
			return GetSceneObject(obj.GetName());
		}

		public GameObject GetSceneObject(string bone) {
			return System.Array.Find(GetSceneObjects(), x => x.name == bone);
		}

		public GameObject[] GetSceneObjects() {
			Scene scene = Asset.GetScene();

			GameObject[] sceneObjects = scene.GetRootGameObjects();
			if(sceneObjects.Length == 0){
				Debug.Log("No GameObjects in Scene found.");
				return null;
			}

			GameObject[] result = new GameObject[0];
			for (int i = 0; i < sceneObjects.Length; i++)
			{
				Transform[] childs = sceneObjects[i].transform.GetAllChilds();
				//check if layermask is in gameobject layer
				childs = childs.Where(x => (ObjectMask & (1 << x.gameObject.layer)) != 0).ToArray();
				result = childs.Select(i => i.gameObject).ToArray();
			}
			return result;
		}

		private void SyncObjects() {
			// Remove objects that are not present in scene
			for (int i = 0; i < InteractableObjects.Length; i++)
			{
				if(InteractableObjects[i] == null) {
					ArrayExtensions.RemoveAt(ref InteractableObjects, i);
				}
				else if(GetSceneObject(InteractableObjects[i]) == null){
					RemoveObject(InteractableObjects[i]);
				}
			}
			// Add objects that present in scene but not in module
			GameObject[] sceneObjects = GetSceneObjects();
			for (int i = 0; i < sceneObjects.Length; i++)
			{
				if(GetObject(sceneObjects[i].name) == null) {
					AddObject(sceneObjects[i].name);
				}
			}
		}

		[System.Serializable]
		public class InteractableObject {
			public IKAugmentationModule Module = null;
			public int Bone = 0;
			public Vector3 DeltaPosition = Vector3.zero;
			public Vector3 DeltaRotation = Vector3.zero;
			public InteractableObject(IKAugmentationModule module, int bone, Vector3 position, Vector3 rotation) {
				Module = module;
				Bone = bone;
				DeltaPosition = position;
				DeltaRotation = rotation;
			}
			public string GetName() {
				return Module.Asset.Source.Bones[Bone].GetName();
			}
			public int GetIndex() {
				return System.Array.FindIndex(Module.InteractableObjects, x => x==this);
			}
			public void SetTransform(float timestamp, bool mirror) {
				Module.GetSceneObject(this).transform.SetTransformation(GetTransformation(timestamp, mirror));
			}
			public Matrix4x4 GetSourceTransformation(float timestamp, bool mirror) {
				return GetTransformation(timestamp, mirror, Matrix4x4.identity);
			}
			public Matrix4x4 GetTransformation(float timestamp, bool mirror) {
				return GetTransformation(timestamp, mirror, Matrix4x4.TRS(DeltaPosition, Quaternion.Euler(DeltaRotation), Vector3.one));
			}
			private Matrix4x4 GetTransformation(float timestamp, bool mirror, Matrix4x4 delta) {
				return Module.Asset.GetFrame(timestamp).GetBoneTransformation(Bone, mirror) * delta;
			}

#if UNITY_EDITOR
			public void Inspector(MotionEditor editor) {
				UltiDraw.Begin();
				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Object", GUILayout.Width(40f));
					EditorGUI.BeginDisabledGroup(true);
					EditorGUILayout.ObjectField(Module.GetSceneObject(GetName()), typeof(GameObject), true);
					EditorGUI.EndDisabledGroup();
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginVertical();
					DeltaPosition = EditorGUILayout.Vector3Field("Delta Position", DeltaPosition);
					DeltaRotation = EditorGUILayout.Vector3Field("Delta Rotation", DeltaRotation);
					EditorGUILayout.EndVertical();
				}
				UltiDraw.End();
			}
#endif
		}
	}
}

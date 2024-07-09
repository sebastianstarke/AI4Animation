﻿#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Xml;
using System;

namespace AI4Animation {
	public class URDFImporter : EditorWindow {

		[MenuItem ("AI4Animation/Importer/URDF Importer")]
		static void Init() {
			EditorWindow window = EditorWindow.GetWindow(typeof (URDFImporter));
			window.minSize = new Vector2(500, 145);
			window.maxSize = new Vector2(500, 145);
		}

		public enum ExportOrientation{YUp, ZUp, Custom};

		private ExportOrientation Export = ExportOrientation.YUp;
		private bool DebugData = false;
		private string Path = string.Empty;
		private Vector3 Orientation = new Vector3(270f, 90f, 0f);
		private string Output = string.Empty;
		private int Errors = 0;

		void OnGUI() {
			SetGUIColor(new Color(0.3f, 0.6f, 0.6f, 1f));
			using(new EditorGUILayout.VerticalScope ("Box")) {

				SetGUIColor(new Color(0.75f, 0.75f, 0.75f));
				EditorGUILayout.LabelField("Path");
				SetGUIColor(new Color(0.75f, 0.75f, 0.75f, 1f));
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();

					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45));
					Path = EditorGUILayout.TextField(Path);
					SetGUIColor(new Color(1f, 1f, 1f));
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						string path = EditorUtility.OpenFilePanel("URDF Importer", Application.dataPath, "urdf");
						if(path.Length != 0) {
							if(path.Contains("Assets/")) {
								Path = path.Substring(path.IndexOf("Assets/")+7);
							} else {
								Debug.Log("Please specify a path inside the Assets folder.");
							}
							GUI.SetNextControlName("");
							GUI.FocusControl("");
						}
					}

					EditorGUILayout.EndHorizontal();
				}

				SetGUIColor(new Color(0.75f, 0.75f, 0.75f));
				EditorGUILayout.LabelField("Mesh Export Orientation");
				SetGUIColor(new Color(0.75f, 0.75f, 0.75f, 1f));
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();

					Export = (ExportOrientation)EditorGUILayout.EnumPopup(Export);
					if(Export == ExportOrientation.YUp) {
						Orientation = new Vector3(0f, 90f, 0f);
					}
					if(Export == ExportOrientation.ZUp) {
						Orientation = new Vector3(270f, 90f, 0f);
					}
					if(Export == ExportOrientation.Custom) {
						Orientation = EditorGUILayout.Vector3Field("", Orientation);
					}

					EditorGUILayout.EndHorizontal();
				}

				SetGUIColor(new Color(1f, 1f, 1f));
				DebugData = EditorGUILayout.Toggle("Debug Data", DebugData);

				SetGUIColor(new Color(1f, 1f, 1f));
				GUI.skin.button.alignment = TextAnchor.MiddleCenter;
				if(GUILayout.Button("Import Model")) {
					Import();
				}
				
			}
		}

		public GameObject Import() {
			if(Application.isPlaying) {
				Debug.Log("Can not import during runtime. Use in Edit-Mode only.");
				return null;
			}
			if(Path == string.Empty) {
				Debug.Log("Path is empty.");
				return null;
			}

			URDFData data = new URDFData(Path);
			if(data.Failed) {
				//Debug.Log("Importing failed.");
				return null;
			} else {
				if(DebugData) {
					Debug.Log("Importing " + Path + "...");
					data.Log();
				}
				return CreateFromData(data);
			}
		}

		private GameObject CreateFromData(URDFData data) {
			Transform actor = new GameObject(data.Name).transform;

			actor.position = new Vector3(0f,0f,0f);
			actor.rotation = Quaternion.identity;

			List<Transform> Links = new List<Transform>();
			List<Transform> Joints = new List<Transform>();

			//Create Link Transforms
			for(int i=0; i<data.Links.Count; i++) {
				Transform link = CreateGeometry(data.Links[i].Visual.Geometry).transform;
				link.name = data.Links[i].Name;
				link.SetParent(actor);
				Links.Add(link);
			}

			//Create Joint Transforms
			for(int i=0; i<data.Joints.Count; i++) {
				Transform joint = new GameObject().transform;
				joint.name = data.Joints[i].Name;
				joint.SetParent(actor);
				Joints.Add(joint);
			}

			//Apply Parent-Child Relations
			for(int i=0; i<Joints.Count; i++) {
				Transform joint = Joints[i];
				Transform parent = FindTransform(Links, data.GetJointData(joint.name).Parent);
				Transform child = FindTransform(Links, data.GetJointData(joint.name).Child);

				Transform parentJoint = actor;
				string parentName = data.GetLinkData(parent.name).Name;
				for(int j=0; j<Joints.Count; j++) {
					if(data.GetJointData(Joints[j].name).Child == parentName) {
						parentJoint = Joints[j];
						break;
					}
				}

				joint.SetParent(parentJoint);
				child.SetParent(joint);
			}

			Links = GetOrderedTransforms(actor.root, Links, new List<Transform>());
			Joints = GetOrderedTransforms(actor.root, Joints, new List<Transform>());

			for(int i=0; i<Joints.Count; i++) {
				Transform joint = Joints[i];
				URDFData.JointData jointData = data.GetJointData(joint.name);
				Vector3 angles = -Mathf.Rad2Deg * ROSToUnity(jointData.OriginRPY);
				Quaternion rotation = Quaternion.Euler(angles);
				joint.position = joint.parent.position + joint.parent.rotation * ROSToUnity(jointData.OriginXYZ);
				joint.rotation = joint.parent.rotation * rotation;
			}
				
			for(int i=0; i<Links.Count; i++) {
				Transform link = Links[i];
				URDFData.LinkData linkData = data.GetLinkData(link.name);
				Vector3 angles = -Mathf.Rad2Deg * ROSToUnity(linkData.Visual.RPY);
				Quaternion rotation = Quaternion.Euler(angles);
				link.localPosition += ROSToUnity(linkData.Visual.XYZ);
				link.localRotation = rotation * link.localRotation;
				
				switch(linkData.Collision.Geometry.Type) {
					case URDFData.ShapeType.Box:
					BoxCollider box = link.gameObject.AddComponent<BoxCollider>();
					box.size = ROSToUnity(((URDFData.Box)linkData.Collision.Geometry).Size);
					box.center = ROSToUnity(linkData.Collision.XYZ);
					break;
					case URDFData.ShapeType.Sphere:
					SphereCollider sphere = link.gameObject.AddComponent<SphereCollider>();
					sphere.radius = ((URDFData.Sphere)linkData.Collision.Geometry).Radius;
					sphere.center = ROSToUnity(linkData.Collision.XYZ);
					break;
					case URDFData.ShapeType.Capsule:
					CapsuleCollider capsule = link.gameObject.AddComponent<CapsuleCollider>();
					capsule.radius = ((URDFData.Capsule)linkData.Collision.Geometry).Radius;
					capsule.height = ((URDFData.Capsule)linkData.Collision.Geometry).Length;
					capsule.center = ROSToUnity(linkData.Collision.XYZ);
					capsule.direction = 0;
					break;
				}
			}

			if(Errors != 0) {
				Debug.Log(Errors + " errors or warnings during importing '" + actor.name + "'.\n\n" + Output);
			}
			Output = string.Empty;
			Errors = 0;

			return actor.gameObject;
		}

		private List<Transform> GetOrderedTransforms(Transform t, List<Transform> inList, List<Transform> outList) {
			if(inList.Contains(t)) {
				outList.Add(t);
			}
			for(int i=0; i<t.childCount; i++) {
				GetOrderedTransforms(t.GetChild(i), inList, outList);
			}
			return outList;
		}

		private GameObject CreatePlaceholder() {
			GameObject placeholder = GameObject.CreatePrimitive(PrimitiveType.Sphere);
			placeholder.transform.localScale = new Vector3(0.025f, 0.025f, 0.025f);
			return placeholder;
		}

		private GameObject CreateGeometry(URDFData.Shape geometryData) {
			GameObject Object = null;
			switch(geometryData.Type) {
			case URDFData.ShapeType.Box:
				URDFData.Box box = (URDFData.Box)geometryData;
				Object = GameObject.CreatePrimitive(PrimitiveType.Cube);
				Object.transform.localScale = ROSToUnity(box.Size);
				break;

			case URDFData.ShapeType.Cylinder:
				URDFData.Cylinder cylinder = (URDFData.Cylinder)geometryData;
				Object = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
				Object.transform.localScale = ROSToUnity(new Vector3(2f*cylinder.Radius, 2f*cylinder.Radius, cylinder.Length/2f));
				break;

			case URDFData.ShapeType.Sphere:
				URDFData.Sphere sphere = (URDFData.Sphere)geometryData;
				Object = GameObject.CreatePrimitive(PrimitiveType.Sphere);
				Object.transform.localScale = ROSToUnity(new Vector3(2f*sphere.Radius, 2f*sphere.Radius, 2f*sphere.Radius));
				break;

			case URDFData.ShapeType.Mesh:
				URDFData.Mesh mesh = (URDFData.Mesh)geometryData;
				GameObject model = AssetDatabase.LoadAssetAtPath(mesh.Path, (typeof(GameObject))) as GameObject;
				if(model != null) {
					Object = Instantiate(model) as GameObject;
					Object.transform.localPosition = Object.transform.localPosition;
					Object.transform.localRotation = Object.transform.localRotation;
					Object.transform.localScale = Vector3.Scale(Object.transform.localScale, mesh.Scale);

					Object.transform.RotateAround(Vector3.zero, Vector3.right, Orientation.x);
					Object.transform.RotateAround(Vector3.zero, Vector3.up, Orientation.y);
					Object.transform.RotateAround(Vector3.zero, Vector3.forward, Orientation.z);

					Object.name = model.name;
				} else {
					Output += "Failed to import mesh '" + mesh.Path + "'.\nThe mesh format must be of .dae or .obj to be imported by Unity. Creating placeholder instead.\n";
					Output += "\n";
					Errors += 1;
					Object = CreatePlaceholder();
				}

				break;

			case URDFData.ShapeType.Empty:
				Object = new GameObject();
				break;
			}

			if(Object.GetComponent<Collider>() != null) {
				DestroyImmediate(Object.GetComponent<Collider>());
			}

			return Object;
		}

		private Transform FindTransform(List<Transform> transforms, string name) {
			return transforms.Find(x => x.name.Equals(name));
		}

		private Vector3 ROSToUnity(Vector3 ROSVec) {
			return new Vector3(-ROSVec.y, ROSVec.z, ROSVec.x);
		}

		private void SetGUIColor(Color color) {
			GUI.backgroundColor = color;
		}

		public class URDFData {

			public string Path;
			public string Folder;
			public bool Failed;
			public string Name;
			public List<MaterialData> Materials;
			public List<JointData> Joints;
			public List<LinkData> Links;

			public URDFData(string path) {
				Failed = false;

				Materials = new List<MaterialData>();
				Links = new List<LinkData>();
				Joints = new List<JointData>();

				Path = path;
				Folder = Path.Substring(0, Path.LastIndexOf("/")+1);

				XmlDocument file = new XmlDocument();
				try {
					file.Load(ConcatPath(Application.dataPath, Path));
				} catch(Exception e) {
					Debug.Log(e.Message);
					Failed = true;
					return;
				}

				XmlNode robotNode = null;
				for(int i=0; i<file.ChildNodes.Count; i++) {
					if(file.ChildNodes[i].Name == "robot") {
						robotNode = file.ChildNodes[i];
						Name = robotNode.Attributes.GetNamedItem("name").Value;
					}
				}

				//Read all first-layer childs of <robot ...>
				for(int i=0; i<robotNode.ChildNodes.Count; i++) {

					//Read Material
					if(robotNode.ChildNodes[i].Name == "material") {
						XmlNode materialNode = robotNode.ChildNodes[i];
						URDFData.MaterialData materialData = new URDFData.MaterialData(materialNode.Attributes.GetNamedItem("name").Value);

						for(int j=0; j<materialNode.ChildNodes.Count; j++) {
							if(materialNode.ChildNodes[j].Name == "color") {
								XmlNode colorNode = materialNode.ChildNodes[j];

								XmlNode rgba = colorNode.Attributes.GetNamedItem("rgba");
								if(rgba != null) {
									materialData.Color = ReadColor(rgba.Value);
								}
							}

							if(materialNode.ChildNodes[j].Name == "texture") {
								XmlNode textureNode = materialNode.ChildNodes[j];

								XmlNode fileName = textureNode.Attributes.GetNamedItem("filename");
								if(fileName != null) {
									materialData.Texture = ConcatPath(Folder, fileName.Value);
								}
							}
						}

						Materials.Add(materialData);
					}

					//Read Link
					if(robotNode.ChildNodes[i].Name == "link") {
						XmlNode linkNode = robotNode.ChildNodes[i];

						//Multiple instances of visual are possible
						string name = linkNode.Attributes.GetNamedItem("name").Value;
						URDFData.LinkData linkData = null;
						if(GetLinkData(name) != null) {
							linkData = GetLinkData(name);
						} else {
							linkData = new URDFData.LinkData(name);
						}

						//Only allow one entry per link
						bool visual = false;
						bool collision = false;

						for(int j=0; j<linkNode.ChildNodes.Count; j++) {
							if(linkNode.ChildNodes[j].Name == "visual" && !visual) {
								visual = true;

								XmlNode node = linkNode.ChildNodes[j];

								for(int k=0; k<node.ChildNodes.Count; k++) {
									if(node.ChildNodes[k].Name == "geometry") {
										XmlNode geometryNode = node.ChildNodes[k];
										
										for(int l=0; l<geometryNode.ChildNodes.Count; l++) {
											if(geometryNode.ChildNodes[l].Name == "box") {
												linkData.Visual.Geometry = new URDFData.Box(ReadVector3(geometryNode.ChildNodes[l].Attributes.GetNamedItem("size").Value));
											} else if(geometryNode.ChildNodes[l].Name == "cylinder") {
												linkData.Visual.Geometry = new URDFData.Cylinder(ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("length").Value), ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("radius").Value));
											} else if(geometryNode.ChildNodes[l].Name == "sphere") {
												linkData.Visual.Geometry = new URDFData.Sphere(ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("radius").Value));
											} else if(geometryNode.ChildNodes[l].Name == "mesh") {
												string meshPath = "Assets/" + ConcatPath(Folder, geometryNode.ChildNodes[l].Attributes.GetNamedItem("filename").Value);
												XmlNode scaleNode = geometryNode.ChildNodes[l].Attributes.GetNamedItem("scale");
												if(scaleNode != null) {
													//toLower
													linkData.Visual.Geometry = new URDFData.Mesh(meshPath, ReadVector3(scaleNode.Value));
												} else {
													//toLower
													linkData.Visual.Geometry = new URDFData.Mesh(meshPath, Vector3.one);
												}
											}
										}
									}

									if(node.ChildNodes[k].Name == "material") {
										XmlNode materialNode = node.ChildNodes[k];

										linkData.Visual.Material = materialNode.Attributes.GetNamedItem("name").Value;

										if(materialNode.HasChildNodes) {
											URDFData.MaterialData materialData = new URDFData.MaterialData(linkData.Visual.Material);

											for(int l=0; l<materialNode.ChildNodes.Count; l++) {
												if(materialNode.ChildNodes[l].Name == "color") {
													XmlNode colorNode = materialNode.ChildNodes[l];

													XmlNode rgba = colorNode.Attributes.GetNamedItem("rgba");
													if(rgba != null) {
														materialData.Color = ReadColor(rgba.Value);
													}
												}

												if(materialNode.ChildNodes[l].Name == "texture") {
													XmlNode textureNode = materialNode.ChildNodes[l];

													XmlNode fileName = textureNode.Attributes.GetNamedItem("filename");
													if(fileName != null) {
														materialData.Texture = ConcatPath(Folder, fileName.Value);
													}
												}
											}

											Materials.Add(materialData);
										}

									}

									if(node.ChildNodes[k].Name == "origin") {
										XmlNode originNode = node.ChildNodes[k];

										XmlNode xyz = originNode.Attributes.GetNamedItem("xyz");
										if(xyz != null) {
											linkData.Visual.XYZ = ReadVector3(xyz.Value);
										}
										XmlNode rpy = originNode.Attributes.GetNamedItem("rpy");
										if(rpy != null) {
											linkData.Visual.RPY = ReadVector3(rpy.Value);
										}
									}
								}
							}
						}

						for(int j=0; j<linkNode.ChildNodes.Count; j++) {
							if(linkNode.ChildNodes[j].Name == "collision" && !collision) {
								collision = true;

								XmlNode node = linkNode.ChildNodes[j];

								for(int k=0; k<node.ChildNodes.Count; k++) {
									if(node.ChildNodes[k].Name == "geometry") {
										XmlNode geometryNode = node.ChildNodes[k];
										
										for(int l=0; l<geometryNode.ChildNodes.Count; l++) {
											if(geometryNode.ChildNodes[l].Name == "box") {
												linkData.Collision.Geometry = new URDFData.Box(ReadVector3(geometryNode.ChildNodes[l].Attributes.GetNamedItem("size").Value));
											} else if(geometryNode.ChildNodes[l].Name == "cylinder") {
												linkData.Collision.Geometry = new URDFData.Cylinder(ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("length").Value), ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("radius").Value));
											} else if(geometryNode.ChildNodes[l].Name == "capsule") {
												linkData.Collision.Geometry = new URDFData.Capsule(ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("length").Value), ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("radius").Value));
											} else if(geometryNode.ChildNodes[l].Name == "sphere") {
												linkData.Collision.Geometry = new URDFData.Sphere(ReadFloat(geometryNode.ChildNodes[l].Attributes.GetNamedItem("radius").Value));
											} else if(geometryNode.ChildNodes[l].Name == "mesh") {
												string meshPath = "Assets/" + ConcatPath(Folder, geometryNode.ChildNodes[l].Attributes.GetNamedItem("filename").Value);
												XmlNode scaleNode = geometryNode.ChildNodes[l].Attributes.GetNamedItem("scale");
												if(scaleNode != null) {
													//toLower
													linkData.Collision.Geometry = new URDFData.Mesh(meshPath, ReadVector3(scaleNode.Value));
												} else {
													//toLower
													linkData.Collision.Geometry = new URDFData.Mesh(meshPath, Vector3.one);
												}
											}
										}
									}

									if(node.ChildNodes[k].Name == "material") {
										XmlNode materialNode = node.ChildNodes[k];

										linkData.Collision.Material = materialNode.Attributes.GetNamedItem("name").Value;

										if(materialNode.HasChildNodes) {
											URDFData.MaterialData materialData = new URDFData.MaterialData(linkData.Collision.Material);

											for(int l=0; l<materialNode.ChildNodes.Count; l++) {
												if(materialNode.ChildNodes[l].Name == "color") {
													XmlNode colorNode = materialNode.ChildNodes[l];

													XmlNode rgba = colorNode.Attributes.GetNamedItem("rgba");
													if(rgba != null) {
														materialData.Color = ReadColor(rgba.Value);
													}
												}

												if(materialNode.ChildNodes[l].Name == "texture") {
													XmlNode textureNode = materialNode.ChildNodes[l];

													XmlNode fileName = textureNode.Attributes.GetNamedItem("filename");
													if(fileName != null) {
														materialData.Texture = ConcatPath(Folder, fileName.Value);
													}
												}
											}

											Materials.Add(materialData);
										}

									}

									if(node.ChildNodes[k].Name == "origin") {
										XmlNode originNode = node.ChildNodes[k];

										XmlNode xyz = originNode.Attributes.GetNamedItem("xyz");
										if(xyz != null) {
											linkData.Collision.XYZ = ReadVector3(xyz.Value);
										}
										XmlNode rpy = originNode.Attributes.GetNamedItem("rpy");
										if(rpy != null) {
											linkData.Collision.RPY = ReadVector3(rpy.Value);
										}
									}
								}
							}
						}

						Links.Add(linkData);
					}

					//Read Joint
					if(robotNode.ChildNodes[i].Name == "joint") {
						XmlNode jointNode = robotNode.ChildNodes[i];
						URDFData.JointData jointData = new URDFData.JointData(jointNode.Attributes.GetNamedItem("name").Value, jointNode.Attributes.GetNamedItem("type").Value);

						for(int j=0; j<jointNode.ChildNodes.Count; j++) {
							if(jointNode.ChildNodes[j].Name == "parent") {
								jointData.Parent = jointNode.ChildNodes[j].Attributes.GetNamedItem("link").Value;
							}
							if(jointNode.ChildNodes[j].Name == "child") {
								jointData.Child = jointNode.ChildNodes[j].Attributes.GetNamedItem("link").Value;
							}
							if(jointNode.ChildNodes[j].Name == "axis") {
								XmlNode xyz = jointNode.ChildNodes[j].Attributes.GetNamedItem("xyz");
								if(xyz != null) {
									jointData.Axis = ReadVector3(xyz.Value);
								}
							}
							if(jointNode.ChildNodes[j].Name == "origin") {
								XmlNode xyz = jointNode.ChildNodes[j].Attributes.GetNamedItem("xyz");
								if(xyz != null) {
									jointData.OriginXYZ = ReadVector3(xyz.Value);
								}
								XmlNode rpy = jointNode.ChildNodes[j].Attributes.GetNamedItem("rpy");
								if(rpy != null) {
									jointData.OriginRPY = ReadVector3(rpy.Value);
								}
							}
							if(jointNode.ChildNodes[j].Name == "limit") {
								XmlNode velocity = jointNode.ChildNodes[j].Attributes.GetNamedItem("velocity");
								if(velocity != null) {
									jointData.Velocity = ReadFloat(velocity.Value);
								}
								XmlNode lower = jointNode.ChildNodes[j].Attributes.GetNamedItem("lower");
								if(lower != null) {
									jointData.LowerLimit = ReadFloat(lower.Value);
								}
								XmlNode upper = jointNode.ChildNodes[j].Attributes.GetNamedItem("upper");
								if(upper != null) {
									jointData.UpperLimit = ReadFloat(upper.Value);
								}
							}
						}
						Joints.Add(jointData);
					}
				}
			}

			public string ConcatPath(string A, string B) {
				if(A.EndsWith("/")) {
					A = A.Substring(0, A.Length-1);;
				}
				while(B.StartsWith("../")) {
					B = B.Substring(3);
					A = A.Substring(0, A.LastIndexOf("/"));
				}
				string C = A + "/" + B;
				C = C.Replace("//", "/");
				return C;
			}

			public MaterialData GetMaterialData(string name) {
				return Materials.Find(x => x.Name.Equals(name));
			}

			public LinkData GetLinkData(string name) {
				return Links.Find(x => x.Name.Equals(name));
			}

			public JointData GetJointData(string name) {
				return Joints.Find(x => x.Name.Equals(name));
			}

			public void Log() {
				Debug.Log("Path: " + Path + "\n");
				Debug.Log("Folder: " + Folder + "\n");

				Debug.Log("Name: " + Name + "\n");
				Debug.Log("/////MATERIALS/////");
				for(int i=0; i<Materials.Count; i++) {
					Debug.Log(
						"Name: " + Materials[i].Name + "\n"
						+	"Color: " + Materials[i].Color + "\n"
						+	"Texture: " + Materials[i].Texture);
				}
				Debug.Log("/////LINKS/////");
				for(int i=0; i<Links.Count; i++) {
					Debug.Log(
						"Name: " + Links[i].Name + "\n"
						+ Links[i].Visual.Format() + "\n"
						+ Links[i].Collision.Format()
					);
				}
				Debug.Log("/////JOINTS/////");
				for(int i=0; i<Joints.Count; i++) {
					Debug.Log(
						"Name: " + Joints[i].Name + "\n"
						+	"Type: " + Joints[i].Type + "\n"
						+	"Parent: " + GetLinkData(Joints[i].Parent).Name + "\n"
						+	"Parent: " + GetLinkData(Joints[i].Child).Name + "\n"
						+	"Origin: " + Joints[i].OriginXYZ.ToString("F4") + "\n"
						+	"RPY: " + Joints[i].OriginRPY.ToString("F4") + "\n"
						+	"Axis: " + Joints[i].Axis.ToString("F4"));
				}
			}

			public class MaterialData {
				public string Name;
				public Color Color;
				public string Texture;

				public MaterialData(string name) {
					Name = name;
					Color = Color.white;
					Texture = string.Empty;
				}
			}

			public class VisualData {
				public Shape Geometry;
				public string Material;
				public Vector3 XYZ;
				public Vector3 RPY;
				public VisualData() {
					Geometry = new Empty();
					Material = string.Empty;
					XYZ = Vector3.zero;
					RPY = Vector3.zero;
				}
				public string Format() {
					return
						"Visual Data\n"
					+	"Geometry Type: " + Geometry.Type + "\n"
					+	Geometry.GetDataInfo() + "\n"
					+	"Material: " + Material + "\n"
					+	"Origin: " + XYZ.ToString("F4") + "\n"
					+	"RPY: " + RPY.ToString("F4");
				}
			}

			public class CollisionData {
				public Shape Geometry;
				public string Material;
				public Vector3 XYZ;
				public Vector3 RPY;
				public CollisionData() {
					Geometry = new Empty();
					Material = string.Empty;
					XYZ = Vector3.zero;
					RPY = Vector3.zero;
				}
				public string Format() {
					return
						"Collision Data\n"
					+	"Geometry Type: " + Geometry.Type + "\n"
					+	Geometry.GetDataInfo() + "\n"
					+	"Material: " + Material + "\n"
					+	"Origin: " + XYZ.ToString("F4") + "\n"
					+	"RPY: " + RPY.ToString("F4");
				}
			}

			public class LinkData {
				public string Name;
				public VisualData Visual;
				public CollisionData Collision;

				public LinkData(string name) {
					Name = name;
					Visual = new VisualData();
					Collision = new CollisionData();
				}
			}

			public class JointData {
				public string Name;
				public string Type;
				public Vector3 OriginXYZ;
				public Vector3 OriginRPY;
				public Vector3 Axis;
				public string Parent;
				public string Child;
				public float Velocity;
				public float LowerLimit;
				public float UpperLimit;

				public JointData(string name, string type) {
					Name = name;
					Type = type;
					OriginXYZ = Vector3.zero;
					OriginRPY = Vector3.zero;
					Axis = new Vector3(1f,0f,0f);
					Parent = string.Empty;
					Child = string.Empty;
					Velocity = 0f;
					LowerLimit = 0f;
					UpperLimit = 0f;
				}
			}

			public enum ShapeType{Box, Cylinder, Capsule, Sphere, Mesh, Empty};

			public abstract class Shape {
				public ShapeType Type;
				public Shape(ShapeType type) {
					Type = type;
				}
				public abstract string GetDataInfo();
			}

			public class Box : Shape {
				public Vector3 Size;
				public Box(Vector3 size) : base(ShapeType.Box) {
					Size = size;
				}
				public override string GetDataInfo() {
					return "Size: " + Size.ToString();
				}
			}

			public class Cylinder : Shape {
				public float Length;
				public float Radius;
				public Cylinder(float length, float radius) : base(ShapeType.Cylinder) {
					Length = length;
					Radius = radius;
				}
				public override string GetDataInfo() {
					return "Length: " + Length + "\n" + "Radius: " + Radius;
				}
			}

			public class Capsule : Shape {
				public float Length;
				public float Radius;
				public Capsule(float length, float radius) : base(ShapeType.Capsule) {
					Length = length;
					Radius = radius;
				}
				public override string GetDataInfo() {
					return "Length: " + Length + "\n" + "Radius: " + Radius;
				}
			}

			public class Sphere : Shape {
				public float Radius;
				public Sphere(float radius) : base(ShapeType.Sphere) {
					Radius = radius;
				}
				public override string GetDataInfo() {
					return "Radius: " + Radius;
				}
			}

			public class Mesh : Shape {
				public string Path;
				public Vector3 Scale;
				public Mesh(string path, Vector3 scale) : base(ShapeType.Mesh) {
					Path = path;
					Scale = scale;
				}
				public override string GetDataInfo() {
					return "Path: " + Path + "\n" + "Scale: " + Scale.ToString("F4");
				}
			}

			public class Empty : Shape {
				public Empty() : base(ShapeType.Empty) {

				}
				public override string GetDataInfo() {
					return "No Geometry";
				}
			}

			private float ReadFloat(string value) {
				value = FilterValueField(value);
				return ParseFloat(value);
			}

			private Vector3 ReadVector3(string value) {
				value = FilterValueField(value);
				string[] values = value.Split(' ');
				float x = ParseFloat(values[0]);
				float y = ParseFloat(values[1]);
				float z = ParseFloat(values[2]);
				return new Vector3(x,y,z);
			}

			private Vector4 ReadColor(string value) {
				value = FilterValueField(value);
				string[] values = value.Split(' ');
				float r = ParseFloat(values[0]);
				float g = ParseFloat(values[1]);
				float b = ParseFloat(values[2]);
				float a = ParseFloat(values[3]);
				return new Color(r,g,b,a);
			}

			private string FilterValueField(string value) {
				while(value.Contains("  ")) {
					value = value.Replace("  "," ");
				}
				while(value.Contains("< ")) {
					value = value.Replace("< ","<");
				}
				while(value.Contains(" >")) {
					value = value.Replace(" >",">");
				}
				while(value.Contains(" .")) {
					value = value.Replace(" ."," 0.");
				}
				while(value.Contains(". ")) {
					value = value.Replace(". ",".0");
				}
				while(value.Contains("<.")) {
					value = value.Replace("<.","<0.");
				}
				return value;
			}

			private float ParseFloat(string value) {
				float parsed = 0f;
				if(float.TryParse(value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out parsed)) {
					return parsed;
				} else {
					Debug.Log("Error parsing " + value + "!");
					return 0f;
				}
			}
		}
	}
}
#endif
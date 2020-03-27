using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace SIGGRAPH_2018 {	

	public class UI : MonoBehaviour {

		public GameObject Character;

		public GameObject Flat;
		public GameObject Terrain;

		public Vector3 BoundsMin = Vector2.zero;
		public Vector3 BoundsMax = Vector2.zero;

		public Button FlatButton;
		public Button TerrainButton;
		public Button SkeletonButton;
		public Button TransformsButton;
		public Button VelocitiesButton;
		public Button TrajectoryButton;
		public Button ExpertActivationButton;
		public Button NetworkWeightsButton;
		public Button FootfallPatternButton;

		public Text EscapeText;

		private bool Skeleton = true;
		private bool Transforms = false;
		private bool Velocities = false;
		private bool Trajectory = true;
		private bool ExpertActivation = false;
		private bool NetworkWeights = false;
		private bool FootfallPattern = false;

		void Start() {
			ApplyVisualisation();
			ApplyGUI();
		}

		void LateUpdate() {
			Vector3 position = Character.transform.position;
			if(position.x < BoundsMin.x || position.z < BoundsMin.z || position.x > BoundsMax.x || position.z > BoundsMax.z) {
				StopAllCoroutines();
				StartCoroutine(Escape());
			}
			if(Input.GetKey(KeyCode.Escape)) {
				Application.Quit();
			}
		}

		private IEnumerator Escape() {
			Character.GetComponent<BioAnimation_Wolf>().Reinitialise();
			float time = 0f;
			float duration = 3f;
			while(time < duration) {
				EscapeText.color = EscapeText.color.Transparent(1f - time / duration);
				time += Time.deltaTime;
				yield return new WaitForEndOfFrame();
			}
			yield return 0;
		}

		void OnDrawGizmos() {
			Gizmos.color = Color.cyan;
			Gizmos.DrawWireCube((BoundsMin+BoundsMax)/2f, BoundsMax - BoundsMin);
		}

		public void LoadFlat() {
			Flat.SetActive(true);
			Terrain.SetActive(false);
			ApplyVisualisation();
			ApplyGUI();
		}

		public void LoadTerrain() {
			Flat.SetActive(false);
			Terrain.SetActive(true);
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleSkeleton() {
			Skeleton = !Skeleton;
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleTransforms() {
			Transforms = !Transforms;
			ApplyVisualisation();
			ApplyGUI();
		}
		
		public void ToggleVelocities() {
			Velocities = !Velocities;
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleTrajectory() {
			Trajectory = !Trajectory;
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleExpertActivation() {
			ExpertActivation = !ExpertActivation;
			NetworkWeights = false;
			FootfallPattern = false;
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleNetworkWeights() {
			ExpertActivation = false;
			NetworkWeights = !NetworkWeights;
			FootfallPattern = false;
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleFootfallPattern() {
			ExpertActivation = false;
			NetworkWeights = false;
			FootfallPattern = !FootfallPattern;
			ApplyVisualisation();
			ApplyGUI();
		}

		private void ApplyVisualisation() {
			Character.GetComponent<ExpertActivation>().enabled = ExpertActivation;
			Character.GetComponent<TensorActivation>().enabled = NetworkWeights;
			Character.GetComponent<FootfallPattern>().enabled = FootfallPattern;
			Character.GetComponent<Actor>().DrawSkeleton = Skeleton;
			Character.GetComponent<Actor>().DrawTransforms = Transforms;
			Character.GetComponent<BioAnimation_Wolf>().ShowVelocities = Velocities;
			Character.GetComponent<BioAnimation_Wolf>().ShowTrajectory = Trajectory;
		}

		private void ApplyGUI() {
			if(Flat.activeSelf) {
				FlatButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				FlatButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if(Terrain.activeSelf) {
				TerrainButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				TerrainButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}

			if(Skeleton) {
				SkeletonButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				SkeletonButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if(Transforms) {
				TransformsButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				TransformsButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if(Velocities) {
				VelocitiesButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				VelocitiesButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if(Trajectory) {
				TrajectoryButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				TrajectoryButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}

			if(ExpertActivation) {
				ExpertActivationButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				ExpertActivationButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if(NetworkWeights) {
				NetworkWeightsButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				NetworkWeightsButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if(FootfallPattern) {
				FootfallPatternButton.GetComponent<Image>().color = UltiDraw.Mustard;
			} else {
				FootfallPatternButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
		}

	}
	
}
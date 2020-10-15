using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace SIGGRAPH_2018 {	

	public class Demo : MonoBehaviour {

		public GameObject Character;

		public GameObject Flat;
		public GameObject Terrain;

		public Button FlatButton;
		public Button TerrainButton;
		public Button SkeletonButton;
		public Button TransformsButton;
		public Button VelocitiesButton;
		public Button TrajectoryButton;
		public Button AuthoringButton;
		public Button LoopButton;
		public Button PausedButton;

		private bool Skeleton = false;
		private bool Transforms = false;
		private bool Velocities = false;
		private bool Trajectory = true;
		private bool Authoring = true;
		private bool Loop = true;
		private bool Paused = true;
		void Start() {
			Loop = Character.GetComponent<Runtime>().AnimationAuthoring.isLooping;
			Authoring = Character.GetComponent<Runtime>().ShowAuthoring;
			Paused = Character.GetComponent<Runtime>().Paused;
			ApplyVisualisation();
			ApplyGUI();
		}

		void OnDrawGizmos() {
			Gizmos.color = Color.cyan;
			//Gizmos.DrawWireCube((BoundsMin+BoundsMax)/2f, BoundsMax - BoundsMin);
		}

		public void LoadFlat() {
			Flat.SetActive(true);
			Terrain.SetActive(false);
			ApplyVisualisation();
			Character.GetComponent<Runtime>().AnimationAuthoring.UpdateLookUpPoints(Character.GetComponent<Runtime>().AnimationAuthoring.TimeDelta);
			ApplyGUI();
		}

		public void LoadTerrain() {
			Flat.SetActive(false);
			Terrain.SetActive(true);
			ApplyVisualisation();
			Character.GetComponent<Runtime>().AnimationAuthoring.UpdateLookUpPoints(Character.GetComponent<Runtime>().AnimationAuthoring.TimeDelta);
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

		public void ToggleAuthoring()
		{
			Authoring = !Authoring;
			ApplyVisualisation();
			ApplyGUI();
		}
		public void TogglePaused()
		{
			Paused = !Paused;
			ApplyVisualisation();
			ApplyGUI();
		}

		public void ToggleLoop()
		{
			Loop = !Loop;
			
			ApplyVisualisation();
			Character.GetComponent<Runtime>().AnimationAuthoring.UpdateLookUpPoints(Character.GetComponent<Runtime>().AnimationAuthoring.TimeDelta);
			ApplyGUI();
		}
		private void ApplyVisualisation() {

			Character.GetComponent<Actor>().DrawSkeleton = Skeleton;
			Character.GetComponent<Actor>().DrawTransforms = Transforms;
			Character.GetComponent<Runtime>().ShowVelocities = Velocities;
			Character.GetComponent<Runtime>().ShowTrajectory = Trajectory;
			Character.GetComponent<Runtime>().ShowAuthoring = Authoring;
			Character.GetComponent<Runtime>().AnimationAuthoring.isLooping = Loop;
			Character.GetComponent<Runtime>().Paused = Paused;
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

			if (Authoring)
			{
				AuthoringButton.GetComponent<Image>().color = UltiDraw.Mustard;
			}
			else
			{
				AuthoringButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}

			if (Loop)
			{
				LoopButton.GetComponent<Image>().color = UltiDraw.Mustard;
			}
			else
			{
				LoopButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
			if (Paused)
			{
				PausedButton.GetComponent<Image>().color = UltiDraw.Mustard;
			}
			else
			{
				PausedButton.GetComponent<Image>().color = UltiDraw.BlackGrey;
			}
		}

	}
	
}
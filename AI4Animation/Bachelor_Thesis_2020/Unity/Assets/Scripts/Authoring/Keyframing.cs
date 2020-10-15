using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(Actor))]
[ExecuteInEditMode]
public class Keyframing : MonoBehaviour
{
	public Actor Start;
	public Actor End;

	[Range(0,1)] public float Slider = 0f;
	public bool Animate = false;

	private Actor Actor;

	 void Awake()
	{

	}
	void Update()
	{

		//Compute Posture
		if(Animate)
		{
			GetActor().Bones[0].Transform.position = Vector3.Lerp(Start.Bones[0].Transform.position, End.Bones[0].Transform.position, Slider);
			for (int i = 0; i < GetActor().Bones.Length; i++)
			{
				GetActor().Bones[i].Transform.rotation = Quaternion.Slerp(Start.Bones[i].Transform.rotation, End.Bones[i].Transform.rotation, Slider);
			}
		}

	}

	private Actor GetActor()
	{
		if (Actor == null)
		{
			Actor = GetComponent<Actor>();
		}
		return Actor;
	}

	

}

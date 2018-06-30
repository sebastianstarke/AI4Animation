using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AnimatorImporter : MonoBehaviour {

	public RuntimeAnimatorController Animator;
	public AnimationClip Animation;

	void OnValidate() {
		/*
		AnimatorOverrideController aoc = new AnimatorOverrideController(GetComponent<Animator>().runtimeAnimatorController);
		var anims = new List<KeyValuePair<AnimationClip, AnimationClip>>();
		foreach (var a in aoc.animationClips)
			anims.Add(new KeyValuePair<AnimationClip, AnimationClip>(a, Animation));
		aoc.ApplyOverrides(anims);
		GetComponent<Animator>().runtimeAnimatorController = aoc;
		GetComponent<Animator>().Play("Animation", 0, 0f);
		transform.position = Vector3.zero;
		transform.rotation = Quaternion.identity;
		*/
	}

	void LateUpdate() {
		//Time.timeScale = 0.5f;
		//GetComponent<Animator>().GetCurrentAnimatorStateInfo()
		//GetComponent<Animator>().Play("Animation");
		//GetComponent<Animator>().PlayInFixedTime("Animation", 0, 0.01f);
		//GetComponent<Animator>().runtimeAnimatorController.animationClips[0].SampleAnimation(gameObject, Timestamp);
		//GetComponent<Animator>().();

		//Timestamp += Time.deltaTime;
	}

}

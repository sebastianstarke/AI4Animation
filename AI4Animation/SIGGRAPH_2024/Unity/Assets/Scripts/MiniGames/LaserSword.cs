using System.Collections;
using UnityEngine;

public class LaserSword : MonoBehaviour
{
    [Space(10)]

    [Header("---------- Laser ----------")]
    [SerializeField] [Range(12, 40)] float laserFadeSpeed = 18;
    [SerializeField] float laserShowTime = 0.3f;
    [SerializeField] float laserHideTime = 0.3f;
    [SerializeField] AnimationCurve curveScaleUp;
    [SerializeField] AnimationCurve curveScaleDown;

    [Header("---------- Flickering ----------")]
    [SerializeField] bool flickering;
    [SerializeField] [Range(0.01f, 0.05f)] float flickRate = 0.025f;
    [SerializeField] [Range(0.1f, 0.9f)] float flickMin = 0.5f;
    [SerializeField] [Range(1, 8)] float lightMultiplier = 2f;
    [SerializeField] Light laserLight;

    [Header("---------- Lens Flare ----------")]
    [SerializeField] LensFlare contactLensFlare;
    [SerializeField] [Range(0, 50)] float lensMin = 50f;
    [SerializeField] [Range(50, 100)] float lensMax = 70f;

    [Header("---------- Particles ----------")]
    [SerializeField] ParticleSystem lightning_particle;
    [SerializeField] ParticleSystem smoke_particle;

    [Header("---------- Audio ----------")]
    [SerializeField] [Range(0.1f, 0.5f)] float hitSoundCooldown = 0.3f;
    [SerializeField] AudioSource audioIdle;
    [SerializeField] AudioSource audioInstant;
    [SerializeField] AudioClip clipHit;
    [SerializeField] AudioClip clipMove;
    [SerializeField] AudioClip clipShowHide;

    private bool reset;
    private int laserLayerIndex;
    private float lerpTime;
    private float currentTime;
    private float flickTimer;
    private float hitAudioTimer;
    private LayerMask laserLayerMask;
    private RaycastHit rHit;

    private Transform rigTrans; //This is used for laser flickering.
    private Transform laserBone; //This is used for laser motion effect (start).
    private Transform boneTarget; //This is used for laser motion effect (end).

    public Transform LaserStart {get; private set;}
    public Transform LaserEnd {get; private set;}
    public Vector3 LaserStartVelocity {get; private set;}
    public Vector3 LaserEndVelocity {get; private set;}
    public Vector3 LaserPosition {get {return (LaserStart.position + LaserEnd.position)/2f;}}
    public Vector3 LaserVelocity {get {return (LaserStartVelocity + LaserEndVelocity)/2f;}}
    private Vector3 PreviousLaserStartPosition;
    private Vector3 PreviousLaserEndPosition;

    #region MONO BEHAVIOUR
    void Start()
    {
        // Caching References
        LaserStart = transform.Find("Laser");
        rigTrans = LaserStart.Find("rig");
        laserBone = rigTrans.Find("Bone03");
        LaserEnd = rigTrans.Find("origin");

        // Set layers
        laserLayerMask = LayerMask.GetMask("LaserSword");
        laserLayerIndex = LayerMask.NameToLayer("LaserSword");

        // Disable
        this.enabled = false;
        rigTrans.localScale = Vector3.zero;

        // Create Target
        boneTarget = new GameObject("LaserPoint").transform;
        boneTarget.position = LaserEnd.position;
        laserBone.localEulerAngles = LaserEnd.localEulerAngles;
        boneTarget.eulerAngles = LaserEnd.eulerAngles;

        //Initialize Previous Positions
        LaserStartVelocity = Vector3.zero;
        LaserEndVelocity = Vector3.zero;
        PreviousLaserStartPosition = LaserStart.position;
        PreviousLaserEndPosition = LaserEnd.position;

        Enable();
    }

    void Update()
    {
        if (!boneTarget)
            return;

        boneTarget.position = Vector3.Lerp(boneTarget.position, LaserEnd.position, lerpTime);
        if (Vector3.Distance(boneTarget.position, LaserEnd.position) > 0.1f)
        {
            reset = false;
            currentTime = 0;
        }

        if (!reset)
        {
            laserBone.right = laserBone.position - boneTarget.position;
            lerpTime = Time.deltaTime * laserFadeSpeed;
        }
        else
        {
            laserBone.localEulerAngles = new Vector3(0, 0, 269.7f);
        }

        currentTime += Time.deltaTime;
        if (currentTime > lerpTime - 0.1f) {
            reset = true;
        }

        //Hit Audio
        hitAudioTimer += Time.deltaTime;
        if (hitAudioTimer > hitSoundCooldown) {
            if (Vector3.Distance(boneTarget.position, LaserEnd.position) > 0.7f)
            {
                hitAudioTimer = 0;

                if (audioInstant != null)
                {
                    audioInstant.clip = clipMove;
                    audioInstant.Play();
                }
            }
        }

        //Update Previous Positions
        LaserStartVelocity = (LaserStart.position - PreviousLaserStartPosition) / Time.smoothDeltaTime;
        LaserEndVelocity = (LaserEnd.position - PreviousLaserEndPosition) / Time.smoothDeltaTime;
        PreviousLaserStartPosition = LaserStart.position;
        PreviousLaserEndPosition = LaserEnd.position;
    }

    void FixedUpdate()
    {
        if (!flickering)
            return;

        flickTimer += Time.deltaTime;
        if (flickTimer > flickRate)
        {
            flickTimer = 0f;
            float r = Random.Range(flickMin, 1f);

            rigTrans.localScale = new Vector3(r * 1f, 1f, r * 1f);

            if (laserLight != null)
                laserLight.intensity = r * lightMultiplier;
        }
    }

    void OnTriggerEnter(Collider coll) {
        // Other Laser
        if (audioInstant != null) {
            audioInstant.clip = clipHit;
            audioInstant.Play();
        }
    }

    void OnTriggerStay(Collider coll) {
        // Other Laser
        if(coll.gameObject.layer == laserLayerIndex) {
            if (Physics.Linecast(LaserStart.position, LaserEnd.position, out rHit, laserLayerMask)) {
                if (contactLensFlare != null) {
                    if (!contactLensFlare.enabled) {
                        contactLensFlare.enabled = true;
                    }
                    contactLensFlare.brightness = Random.Range(lensMin, lensMax);
                    contactLensFlare.transform.position = rHit.point;
                }
            }
        }
    }

    void OnTriggerExit(Collider coll) {
        // Other Laser
        if (coll.gameObject.layer == laserLayerIndex)
        {
            if (contactLensFlare != null)
                contactLensFlare.enabled = false;
        }
    }

    #endregion


    /// <summary>
    /// Enable the laser.
    /// </summary>
    public void Enable()
    {
        if (this.enabled)
            return;

        this.enabled = true;

        boneTarget.position = LaserEnd.position;

        StartCoroutine(IE_ScaleLaser(laserShowTime, curveScaleUp));

        hitAudioTimer = 0;

        #region EFFECTS

        if (audioIdle != null)
            audioIdle.Play();
        if (audioInstant != null)
        {
            audioInstant.clip = clipShowHide;
            audioInstant.Play();
        }

        if (laserLight != null)
            laserLight.enabled = true;
        if (lightning_particle != null && lightning_particle.gameObject.activeSelf == true)
            lightning_particle.Play();
        if (smoke_particle != null && smoke_particle.gameObject.activeSelf == true)
            smoke_particle.Play();

        #endregion
    }

    /// <summary>
    /// Disable the laser.
    /// </summary>
    public void Disable()
    {
        if (!this.enabled)
            return;

        this.enabled = false;

        StartCoroutine(IE_ScaleLaser(laserHideTime, curveScaleDown));


        #region EFFECTS

        if (audioIdle != null)
            audioIdle.Stop();
        if (audioInstant != null)
        {
            audioInstant.clip = clipShowHide;
            audioInstant.Play();
        }

        if (laserLight != null)
            laserLight.enabled = false;

        if (lightning_particle != null && lightning_particle.gameObject.activeSelf == true)
            lightning_particle.Stop();

        if (smoke_particle != null && smoke_particle.gameObject.activeSelf == true)
            smoke_particle.Stop();

        if (contactLensFlare != null)
            contactLensFlare.enabled = false;

        #endregion
    }

    private IEnumerator IE_ScaleLaser(float duration, AnimationCurve curve)
    {
        float timer = 0f;
        while (timer < duration)
        {
            timer += Time.deltaTime;
            timer = Mathf.Clamp(timer, 0f, duration);
            float percent = timer / duration;
            float value = curve.Evaluate(percent);
            rigTrans.localScale = Vector3.one * value;

            yield return null;
        }
    }
}
let model = tf.loadLayersModel('/jsmodel/model.json');
let video = document.getElementById("cam_input"); // video is the id of video tag

function cameraStart() {
    navigator.mediaDevices
        .getUserMedia(configList[idConfig])
        .then(function(stream) {
        // track = stream.getTracks()[0];
        video.srcObject = stream;
        video.play();
    })
    .catch(function(error) {
        console.error("Oops. Something is broken.", error);
    });
}

function openCvReady() {
    cv['onRuntimeInitialized']=()=>{
    
    cameraStart();
    video.style.display="none";
    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(cam_input);
    let faces = new cv.RectVector();
    let classifier = new cv.CascadeClassifier();
    let utils = new Utils('errorMessage');
    let faceCascadeFile = 'haarcascade_frontalface_default.xml'; // path to xml
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
    classifier.load(faceCascadeFile); // in the callback, load the cascade from file 
    });
    const FPS = 15;
    let faceroi = new cv.Mat();
    let idCol = 0;
    const colors = {0: [0, 255, 0, 0], 1:[255, 0, 0, 255]};
    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        // src.copyTo(dst);
        cv.cvtColor(src, dst, cv.COLOR_RGBA2RGB);//ngilangin alpha
        cv.cvtColor(dst, gray, cv.COLOR_RGB2GRAY);//ke item putih
        // cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        try{
            classifier.detectMultiScale(gray, faces, 1.1, 3);
            console.log(faces.size());
        }catch(err){
            console.log(err);
        }
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            // cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
            let rect = new cv.Rect(face.x, face.y, face.width, face.height);
            faceroi = dst.roi(rect);
            //normalisasi (1/255)
            let normal = new Float32Array(faceroi.data.length);
            for(let i = 0;i<faceroi.data.length;i++){
                normal[i] = faceroi.data[i]/255.0;
                // console.log(i +": " + normal[i]);
            }
            let tensor = tf.tensor(normal, [faceroi.rows, faceroi.cols, 3]);
            let resized = tf.image.resizeBilinear(tensor, [128, 128]);
            let finished = resized.reshape([1, 128, 128, 3]);
            
            //predict
            model.then(function(res){
                let result = res.predict(finished);
                let masked = result.dataSync()[0];
                let nonmask = result.dataSync()[1];
                // console.log(masked + " vs " + nonmask);
                idCol = nonmask > masked? 1: 0;

            },function(err){
                console.log("got error:" + err);
            });            
            cv.rectangle(dst, point1, point2, colors[idCol], 2);
        }
        cv.imshow("canvasOutput", dst);
          // schedule next one.
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }
  // schedule first one.
    setTimeout(processVideo, 0);
    };
}
// Set constraints for the video stream
const constraints0 = { video: { facingMode: "user" }, audio: false };
const constraints1 = { video: { facingMode: "environment" }, audio: false };
const configList = [constraints0, constraints1];
var idConfig = 0;
const changeCamBtn = document.querySelector("#change-cam");
if(!!changeCamBtn){
    changeCamBtn.addEventListener('click', function(){
        idConfig++;
        idConfig %= 2;
        alert("change camera");
        cameraStart();

        // shootFirstFrameHandler();

    });
}
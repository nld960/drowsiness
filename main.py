from IPython.display import display, Javascript, Image, Audio
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import time
import dlib
from math import hypot
from google.colab import output


def js_to_image(js_reply):
    """
    convert the JavaScript object into an OpenCV image

    Params:
            js_reply: JavaScript object containing image from webcam
    Returns:
            img: OpenCV BGR image
    """
    # decode base64 image
    image_bytes = b64decode(js_reply.split(",")[1])

    # convert bytes to numpy array
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)

    # decode numpy array into OpenCV BGR image
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


def bbox_to_bytes(bbox_array):
    """
    convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video

    Params:
            bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
    Returns:
          bytes: Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, "RGBA")
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format="png")
    # format return string
    bbox_bytes = "data:image/png;base64,{}".format(
        (str(b64encode(iobuf.getvalue()), "utf-8"))
    )

    return bbox_bytes


# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
    js = Javascript("""
        var video;
        var div = null;
        var stream;
        var captureCanvas;
        var imgElement;
        var labelElement;
        
        var pendingResolve = null;
        var shutdown = false;
        
        function removeDom() {
            stream.getVideoTracks()[0].stop();
            video.remove();
            div.remove();
            video = null;
            div = null;
            stream = null;
            imgElement = null;
            captureCanvas = null;
            labelElement = null;
        }
        
        function onAnimationFrame() {
            if (!shutdown) {
                window.requestAnimationFrame(onAnimationFrame);
            }
            if (pendingResolve) {
                var result = "";
                if (!shutdown) {
                captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                result = captureCanvas.toDataURL('image/jpeg', 0.8)
                }
                var lp = pendingResolve;
                pendingResolve = null;
                lp(result);
            }
        }
        
        async function createDom() {
            if (div !== null) {
                return stream;
            }

            div = document.createElement('div');
            div.style.border = '2px solid black';
            div.style.padding = '3px';
            div.style.width = '100%';
            div.style.maxWidth = '600px';
            document.body.appendChild(div);
            
            const modelOut = document.createElement('div');
            modelOut.innerHTML = "<span>Status:</span>";
            labelElement = document.createElement('span');
            labelElement.innerText = 'No data';
            labelElement.style.fontWeight = 'bold';
            modelOut.appendChild(labelElement);
            div.appendChild(modelOut);
                
            video = document.createElement('video');
            video.style.display = 'block';
            video.width = div.clientWidth - 6;
            video.setAttribute('playsinline', '');
            video.onclick = () => { shutdown = true; };
            stream = await navigator.mediaDevices.getUserMedia(
                {video: { facingMode: "environment"}});
            div.appendChild(video);

            imgElement = document.createElement('img');
            imgElement.style.position = 'absolute';
            imgElement.style.zIndex = 1;
            imgElement.onclick = () => { shutdown = true; };
            div.appendChild(imgElement);
            
            const instruction = document.createElement('div');
            instruction.innerHTML = 
                '<span style="color: red; font-weight: bold;">' +
                'When finished, click here or on the video to stop this webcam</span>';
            div.appendChild(instruction);
            instruction.onclick = () => { shutdown = true; };
            
            video.srcObject = stream;
            await video.play();

            captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640; //video.videoWidth;
            captureCanvas.height = 480; //video.videoHeight;
            window.requestAnimationFrame(onAnimationFrame);
            
            return stream;
        }
        async function stream_frame(label, imgData) {
            if (shutdown) {
                removeDom();
                shutdown = false;
                return '';
            }

            var preCreate = Date.now();
            stream = await createDom();
            
            var preShow = Date.now();
            if (label != "") {
                labelElement.innerHTML = label;
            }
                    
            if (imgData != "") {
                var videoRect = video.getClientRects()[0];
                imgElement.style.top = videoRect.top + "px";
                imgElement.style.left = videoRect.left + "px";
                imgElement.style.width = videoRect.width + "px";
                imgElement.style.height = videoRect.height + "px";
                imgElement.src = imgData;
            }
            
            var preCapture = Date.now();
            var result = await new Promise(function(resolve, reject) {
                pendingResolve = resolve;
            });
            shutdown = false;
            
            return {'create': preShow - preCreate, 
                    'show': preCapture - preShow, 
                    'capture': Date.now() - preCapture,
                    'img': result};
        }
    """
    )

    display(js)


def video_frame(label, bbox):
    data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
    return data



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(  # with mounted Drive in Google Colab
    "/content/drive/MyDrive/shape_predictor_68_face_landmarks.dat"
)


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_TRIPLEX


def compute_blinking_ratio(eye_points, facial_landmarks):
    left_point = (
        facial_landmarks.part(eye_points[0]).x,
        facial_landmarks.part(eye_points[0]).y,
    )
    right_point = (
        facial_landmarks.part(eye_points[3]).x,
        facial_landmarks.part(eye_points[3]).y,
    )
    center_top = midpoint(
        facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2])
    )
    center_bottom = midpoint(
        facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4])
    )

    hor_line_lenght = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1])
    )
    ver_line_lenght = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])
    )

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def compute_mouth_ratio(lips_points, facial_landmarks):
    left_point = (
        facial_landmarks.part(lips_points[0]).x,
        facial_landmarks.part(lips_points[0]).y,
    )
    right_point = (
        facial_landmarks.part(lips_points[2]).x,
        facial_landmarks.part(lips_points[2]).y,
    )
    center_top = (
        facial_landmarks.part(lips_points[1]).x,
        facial_landmarks.part(lips_points[1]).y,
    )
    center_bottom = (
        facial_landmarks.part(lips_points[3]).x,
        facial_landmarks.part(lips_points[3]).y,
    )

    horizontal_line_length = hypot(
        (left_point[0] - right_point[0]), (left_point[1] - right_point[1])
    )
    vertical_line_length = hypot(
        (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1])
    )
    if horizontal_line_length == 0:
        return vertical_line_length
    ratio = vertical_line_length / horizontal_line_length
    return ratio


# start streaming video from webcam
video_stream()

# initialze bounding box to empty
bbox = ""
count = 0
while True:
    js_reply = video_frame("Webcam Capturing...", bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    img = js_to_image(js_reply["img"])

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = detector(gray)

    for face in faces:

        landmarks = predictor(gray, face)

        left_eye_ratio = compute_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = compute_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        # cv2.putText(img, str(blinking_ratio), (0, 13), font, 0.5, (100, 100, 100))
        # print(left_eye_ratio,right_eye_ratio,blinking_ratio)
        # print("1")
        inner_lip_ratio = compute_mouth_ratio([60, 62, 64, 66], landmarks)
        outter_lip_ratio = compute_mouth_ratio([48, 51, 54, 57], landmarks)
        mouth_opening_ratio = (inner_lip_ratio + outter_lip_ratio) / 2
        # cv2.putText(img, str(mouth_opening_ratio), (448, 13), font, 0.5, (100, 100, 100))
        #  print("2")
        #  print(inner_lip_ratio,outter_lip_ratio,mouth_opening_ratio)
        if (
            mouth_opening_ratio > 0.4 and blinking_ratio > 4 or blinking_ratio > 4.3
        ):  # if mouth_opening_ratio > 0.38 and blinking_ratio > 4 or blinking_ratio > 4.3
            time.sleep(2)
            if (
                mouth_opening_ratio > 0.4 and blinking_ratio > 4 or blinking_ratio > 4.3
            ):  # if mouth_opening_ratio > 0.38 and blinking_ratio > 4 or blinking_ratio > 4.3
                time.sleep(2)
                if (
                    mouth_opening_ratio > 0.4
                    and blinking_ratio > 4
                    or blinking_ratio > 4.3
                ):  # if mouth_opening_ratio > 0.38 and blinking_ratio > 4 or blinking_ratio > 4.3
                    count = count + 1

        # else:
        #    count = 0
        left, top = face.left(), face.top()
        right, bottom = face.right(), face.bottom()
        
        if count > 3: 
            bbox_array = cv2.rectangle(bbox_array, (left, top), (right, bottom), (255, 0, 0), 2)
            label_html = "Drowsy"
            Audio(url="/content/drive/MyDrive/Buzzer.wav")
            # output.eval_js('new Audio("/content/drive/MyDrive/Buzzer.wav").play()')
            count = count % 3
        else:
            label_html = "Normal"
            bbox_array = cv2.rectangle(bbox_array, (left, top), (right, bottom), (0, 255, 0), 2)

        # get face bounding box for overlay
        
    bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
    
    # convert overlay of bbox into bytes
    bbox_bytes = bbox_to_bytes(bbox_array)
    
    # update bbox so next frame gets new overlay
    bbox = bbox_bytes

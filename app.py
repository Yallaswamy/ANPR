from flask import *
import cv2
import pytesseract

app=Flask(__name__)

def NPR_gen_frames():
    cap=cv2.VideoCapture(0)
    harcascade = "Haarcascades/haarcascade_russian_plate_number.xml"
    cap.set(3, 640) # width
    cap.set(4, 480) #height
    min_area = 500
    count = 0
    while True:
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            plate_cascade = cv2.CascadeClassifier(harcascade)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

            for (x,y,w,h) in plates:
                area = w * h

                if area > min_area:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    img_roi = img[y: y+h, x:x+w]
                    cv2.imshow("ROI", img_roi)
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def prediction1(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number_plate_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if aspect_ratio > 2.0 and aspect_ratio < 5.0 and area > 1000:
            number_plate_contours.append(contour)
    pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
   # pytesseract.pytesseract.tesseract_cmd = r'tesseract'
    for contour in number_plate_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        roi = blurred[y:y+h, x:x+w]
        number_plate_text = pytesseract.image_to_string(roi, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng')
        return number_plate_text
    return ''


@app.route('/')
def hello():
    return render_template('main.html')

@app.route('/NPR')
def NPR():
    return render_template('number_plate.html')

@app.route('/NPRImage')
def NPRImage():
    return render_template('number_plate_image.html')

    return render_template('text_extraction.html')

@app.route('/predict1',methods=["GET","POST"])
def predict1():
    file=request.files['file']
    file_path = r"static/Storage" + file.filename
    file.save(file_path)  
    k=prediction1(file_path)
    return render_template('number_plate_image.html',ans=k)

@app.route('/NPRVideoLoad')
def NPRVideoLoad():
    return render_template('number_plate_video.html')

@app.route('/NPRVideo')
def NPRVideo():
   return Response(NPR_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True,ssl_context='adhoc')

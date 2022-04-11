import cv2
import face_recognition as face

shajarian01_file = 'face/shajarian1.jpg'
two_face         = 'face/fatherandson.jpg'
win_name         = "shajarian basic image"

img_shajarian    = face.load_image_file(shajarian01_file)
face_location    = face.face_locations(img_shajarian)[0]

point1: tuple    = (face_location[3],face_location[0])
point2: tuple    =  (face_location[1],face_location[2])

img_shajarian    = cv2.rectangle(img_shajarian, point1, point2, (0,255,0),2)

img_shajarian   = cv2.cvtColor(img_shajarian, cv2.COLOR_RGB2BGR)
cv2.imshow(win_name, img_shajarian)
cv2.waitKey(0)
cv2.destroyAllWindows()

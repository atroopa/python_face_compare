import cv2
import face_recognition as face

shajarian01_file = 'face/shajarian1.jpg'
two_face         = 'face/fatherandson.jpg'
win_name         = "shajarian basic image"

two_face_read    = face.load_image_file(two_face)
face_1_location  = face.face_locations(two_face_read, model='hog')[0]
face_2_location  = face.face_locations(two_face_read, model='hog')[1]
print(face_1_location)
print(face_2_location)

face1_point1: tuple    = (face_1_location[3], face_1_location[0])
face1_point2: tuple    = (face_1_location[1], face_1_location[2])

face2_point1: tuple    = (face_2_location[3], face_2_location[0])
face2_point2: tuple    = (face_2_location[1], face_2_location[2])

two_face_read = cv2.rectangle(two_face_read, face1_point1, face1_point2, (0,255,0),2) # Homayon Left
two_face_read = cv2.rectangle(two_face_read, face2_point1, face2_point2, (0,255,0),2) # Mohammad Right

two_face_read    = cv2.cvtColor(two_face_read, cv2.COLOR_RGB2BGR)
cv2.imshow(win_name, two_face_read)
cv2.waitKey(0)
cv2.destroyAllWindows()

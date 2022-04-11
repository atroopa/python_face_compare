import cv2
import face_recognition

# Trainig Image ------------------------------------------

shajarian01_file    = 'face/shajarian1.jpg'
img_shajarian       = face_recognition.load_image_file(shajarian01_file)
face_location       = face_recognition.face_locations(img_shajarian)[0]

point1: tuple       = (face_location[3],face_location[0])
point2: tuple       =  (face_location[1],face_location[2])
img_shajarian       = cv2.rectangle(img_shajarian, point1, point2, (0,255,0),2)

shajarian_face_code = face_recognition.face_encodings(img_shajarian)[0]


# Trainig Image 2------------------------------------------

takhti_file                = 'face/takhti.jpg'
img_takhti                 = face_recognition.load_image_file(takhti_file)
takhti_face_location       = face_recognition.face_locations(img_takhti)[0]

takhti_point1: tuple       = (takhti_face_location[3],takhti_face_location[0])
takhti_point2: tuple       =  (takhti_face_location[1],takhti_face_location[2])
img_takhti                 = cv2.rectangle(img_takhti, takhti_point1, takhti_point2, (0,255,0),2)

takhti_face_code           = face_recognition.face_encodings(img_takhti)[0]


# Region Test Image ----------------------------------------

test_file           = 'face/shajarian2.jpg'
imgTest             = face_recognition.load_image_file(test_file)
test_face_location  = face_recognition.face_locations(imgTest)[0]

test_point_1:tuple  = (test_face_location[3], test_face_location[0])
test_point_2:tuple  = (test_face_location[1], test_face_location[2])
test_face_rectangle = cv2.rectangle(imgTest, test_point_1, test_point_2, (0,255,0),2)

test_face_code      = face_recognition.face_encodings(imgTest)[0]


# Proccessing
isSame              = face_recognition.compare_faces([shajarian_face_code,takhti_face_code],test_face_code)
distance            = face_recognition.face_distance([shajarian_face_code,takhti_face_code],test_face_code)



# Image Show
img_shajarian       = cv2.cvtColor(img_shajarian, cv2.COLOR_RGB2BGR)
test_face_rectangle = cv2.cvtColor(test_face_rectangle, cv2.COLOR_RGB2BGR)

cv2.putText(test_face_rectangle, f'{isSame[0]} {round(distance[0],2)}', (0,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),2)
cv2.imshow("Image in Our DataBase" , img_shajarian)
cv2.imshow("Image2 in Our DataBase" , img_takhti)
cv2.imshow("New Image From User" , test_face_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
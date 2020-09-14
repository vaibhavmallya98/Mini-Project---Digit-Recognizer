import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

draw = False 
a,b = -1,-1
def draw_circle(event,x,y,flags,param):
    global a,b,draw

    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        a,b = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            cv2.circle(img,(x,y),10,(255,0,0),-1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        cv2.circle(img,(x,y),10,(255,0,0),-1)

img = np.zeros([400,400],dtype='uint8')*255
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)



while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    if k == ord('s'):
        cv2.imwrite('Output.jpg',img)
    if k == ord('p'):
        img_pred = cv2.imread('Output.jpg')
        img_resize = cv2.resize(img_pred,(28,28))#.reshape(1,28,28)
        model = tf.keras.models.load_model('MNIST_digits.h5')
        img_gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)
        y_pred = model.predict_classes(img_gray.reshape(1,28,28,1))
        print(y_pred)

cv2.destroyAllWindows()

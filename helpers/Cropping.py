#class to handle cropping options
class Cut:   
    #function to crop image     
    #Parameters: image to crop, contour, and the image number     
    def crop(image, contours, num):          
        idNum = 0          
        #cycles through all the contours to crop all          
        for c in countours:         
            #creates an approximate rectangle around contour          
            x,y,w,h = cv2.boundingRect(cntr)         
            # Only crop decently large rectangles         
            if w>50 and h>50:              
                idNum+=1              
                #pulls crop out of the image based on dimensions              
                new_img=image[y:y+h,x:x+w]              
                #writes the new file in the Crops folder              
                cv2.imwrite('Crops/'+'crop_'+str(num)+'_'+str(idNum)+ '.png', new_img)         
                                                   
            #returns a number incremented up for the next file name         
            return num+1
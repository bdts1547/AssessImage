import sys
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# create SIFT object ((a feature detection algorithm)) 
sift = cv2.xfeatures2d.SIFT_create()
# create BFMatcher object
bf = cv2.BFMatcher()

####################
## Run whole process
####################

def detecting_mirrorLine(picture_name: str, show_detail = False):
    """
    Main function
    
    If show_detail = True, plot matching details 
    """
    # create mirror object
    mirror = Mirror_Symmetry_detection(picture_name)

    # extracting and Matching a pair of symmetric features
    matchpoints = mirror.find_matchpoints()

    # get r, tehta (polar coordinates) of the midpoints of all pair of symmetric features
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)
   
    if show_detail: # visualize process in detail
        mirror.draw_matches(matchpoints, top = 10)
        mirror.draw_hex(points_r, points_theta)

    # find the best one with highest vote
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap= plt.cm.Spectral_r) 
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)  
    
    # add mirror line based on r and theta
    # is_sym = mirror.is_symmetry(r, theta)

    return r, theta



#############################
## Mirror symmetry detection
#############################


class Mirror_Symmetry_detection:
    def __init__(self, image_path: str):
        self.image = self._read_color_image(image_path) # convert the image into the array/matrix

        self.filename = image_path.split('/')[-1]

        self.reflected_image = np.fliplr(self.image) # Flipped version of image 
        
        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None) 
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)
     
    
    def _read_color_image(self, image_path):
        """
        convert the image into the array/matrix with oroginal color
        """
        image = cv2.imread(image_path) # convert the image into the array/matrix
        b,g,r = cv2.split(image)       # get b,g,r
        image = cv2.merge([r,g,b])     # switch it to rgb
        
        return image
        
        
    def find_matchpoints(self):
        """
        Extracting and Matching a pair of symmetric features
    
        Matches are then sort between the features ki and the mirrored features mj 
        to form a set of (pi,pj) pairs of potentially symmetric features. 
    
        Ideally a keypoint at a certain spot on the object in original image should have a descriptor very similar to 
        the descriptor on a point on the object in its mirrored version
        """
        # use BFMatcher.knnMatch() to get （k=2）matches
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        # these matches are equivalent only one need be recorded
        matchpoints = [item[0] for item in matches] 
        
        # sort to determine the dominant symmetries
        # Distance between descriptors. The lower, the better it is.
        matchpoints = sorted(matchpoints, key = lambda x: x.distance) 
        
        return matchpoints
    
    
    def find_points_r_theta(self, matchpoints:list):
        """
        Get r, tehta of the midpoints of all pair of symmetric features
        """
        points_r = [] # list of r for each point
        points_theta = [] # list of theta for each point
        for match in matchpoints:
        
            point = self.kp1[match.queryIdx]  # queryIdx is an index into one set of keypoints, (origin image)
            mirpoint = self.kp2[match.trainIdx] # trainIdx is an index into the other set of keypoints (fliped image)
            
            mirpoint.angle = np.deg2rad(mirpoint.angle) # Normalise orientation
            mirpoint.angle = np.pi - mirpoint.angle
            # convert angles to positive 
            if mirpoint.angle < 0.0:   
                mirpoint.angle += 2*np.pi
                
            # pt: coordinates of the keypoints x:pt[0], y:pt[1]
            # change x, not y
            mirpoint.pt = (self.reflected_image.shape[1]-mirpoint.pt[0], mirpoint.pt[1]) 
                
            # get θij: the angle this line subtends with the x-axis.
            theta = angle_with_x_axis(point.pt, mirpoint.pt)  
            
            # midpoit (xc,yc) are the image centred co-ordinates of the mid-point of the line joining pi and pj
            xc, yc = midpoint(point.pt, mirpoint.pt) 
            r = xc*np.cos(theta) + yc*np.sin(theta)  
    
            points_r.append(r)
            points_theta.append(theta)
            
        return points_r, points_theta # polar coordinates


    def draw_matches(self, matchpoints, top=10):
        """visualize the best matchs
        """
        img = cv2.drawMatches(self.image, self.kp1, self.reflected_image, self.kp2, 
                               matchpoints[:top], None, flags=2) 
        plt.imshow(img); 
        plt.title("Top {} pairs of symmetry points".format(top))
        plt.show() 
        
    def draw_hex(self, points_r: list, points_theta: list):
        """
        Visualize hex bins based on r and theta
        """  
        # Make a 2D hexagonal binning plot of points r and theta 
        image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap= plt.cm.Spectral_r) 
        plt.colorbar() # add color bar
        plt.show()
    
    
    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        """Try to find the x and y coordinates of the hexbin with max count
        """
        for k, v in sorted_vote.items():
            # if mirror line is vertical, return the highest vote
            if vertical:
                return k[0], k[1]
            # otherwise, return the highest vote, whose y is not 0 or pi
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]
            
    
    def sort_hexbin_by_votes(self, image_hexbin):
        """Sort hexbins by decreasing count. (lower vote)
        """
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10,counts)) # get non-zero hexbins
        verts = image_hexbin.get_offsets() # coordinates of each hexbin
        output = {}
        
        for offc in range(verts.shape[0]):
            binx,biny = verts[offc][0],verts[offc][1]
            if counts[offc]:
                output[(binx,biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
                              
    def is_symmetry(self, r, theta): 

        """
        Draw mirror line based on r theta polar co-ordinate
        """
       
        def cal_angle_with(point_1, point_2, axis=None): # y1, y2 ~ min height, max height
            # a.b = |a||b|cos(a)
            x1, y1 = point_1
            x2, y2 = point_2

            
            if axis=='Oy':  # vector relative to the vertical axis
                vector_sym = [x2-x1, y2-y1]
                vector_cmp = [0, y2]
            
            elif axis=='Ox': # vector relative to the horizontal axis
                vector_sym = [x2-x1, y2-y1]
                vector_cmp = [x2, 0]
            else:
                print("Error: Must parameter 'axis' = Ox or Oy")
                return

            unit_vector_sym = vector_sym / np.linalg.norm(vector_sym)
            unit_vector_cmp = vector_cmp / np.linalg.norm(vector_cmp)
            dot_product = np.dot(unit_vector_sym, unit_vector_cmp)
            angle = np.arccos(dot_product)
            return np.degrees(angle)
        
        def is_x_within(x, x_min, x_max):
            if (x > x_min and x < x_max):
                return True
            return False
        
        def detect_symmetric(degree, point_1, point_2, thresh_y = 15, rate=0.2):
            # symmetry through Oy
            h, w, c = self.image.shape

            x1, y1 = point_1
            x2, y2 = point_2

            x_center_min = int(w // 2 - (rate/2 * w))
            x_center_max = int(w // 2 + (rate/2 * w))

            is_within_center = is_x_within(x1, x_center_min, x_center_max) and is_x_within(x2, x_center_min, x_center_max)
            if (degree < thresh_y) and is_within_center:
                return True
            else:
                return False

        
        
        h, w, c = self.image.shape

        # Start,End point
        point_1 = (int((r-0*np.sin(theta))/np.cos(theta)), 0) # (x, y_min)
        point_2 = (int((r-(h-1)*np.sin(theta))/np.cos(theta)), h-1) # (x, y_max)

        # point_1_Ox = (0, int(r / np.sin(theta))) # (x_min, y)
        # point_2_Ox = (w-1, int((r-(w-1)*np.cos(theta)) / np.sin(theta))) # (x_max, y)


        degree = cal_angle_with(point_1, point_2, axis='Oy')
        # print(point_1, '|', point_2)
        # print("Degree:", degree)
        _is_symmetry = detect_symmetric(degree, point_1, point_2)
        
    

        # draw plot 
        cv2.line(self.image, point_1, point_2, (0,0,255), 3)
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(self.image)
        # ax.imshow(self.image)
        fig.savefig('layout/symmetry/{}'.format(self.filename))

        return _is_symmetry

        

def angle_with_x_axis(pi, pj):  # 公式在文件里解释
    """
    calculate θij:
        the angle this line subtends with the x-axis.
    """
    # get the difference between point p1 and p2
    x, y = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

def midpoint(pi, pj):
    """
    get x and y coordinates of the midpoint of pi and pj
    """
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2
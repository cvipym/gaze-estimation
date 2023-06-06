import math

def get_features(mesh_coords):   
    def euclaideanDistance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
      
    L_center = (mesh_coords[33] + mesh_coords[133])/2
    R_center = (mesh_coords[362] + mesh_coords[263])/2
    
    LCen2Iris_x, LCen2Iris_y = mesh_coords[468] - L_center
    RCen2Iris_x, RCen2Iris_y = mesh_coords[473] - R_center

    L_height = euclaideanDistance(mesh_coords[159], mesh_coords[145])
    L_width = euclaideanDistance(mesh_coords[33], mesh_coords[133])
    
    R_height = euclaideanDistance(mesh_coords[386], mesh_coords[374])
    R_width = euclaideanDistance(mesh_coords[362], mesh_coords[263])
    
    
    L_weight = L_width/(L_width+R_width)
    R_weight = R_width/(L_width+R_width)
    
    distance_correction = max(L_width, R_width)    
    
    xi = (LCen2Iris_x*L_weight + RCen2Iris_x*R_weight)/distance_correction
    yi = (LCen2Iris_y*L_weight + RCen2Iris_y*R_weight)/distance_correction
    yl = (L_height*L_weight+R_height*R_weight)/distance_correction
    
    l = euclaideanDistance(mesh_coords[133], mesh_coords[362])
    cos_theta = (mesh_coords[362][0]-mesh_coords[133][0])/l
    sin_theta = (mesh_coords[362][1]-mesh_coords[133][1])/l
    
    xi_hat = xi*cos_theta + yi*sin_theta
    yi_hat = -xi*sin_theta + yi*cos_theta

    return xi_hat, yi_hat, yl   
         
    
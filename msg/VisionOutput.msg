#Centerline[] centerlines       # Endpoints of the center lines in real world
#BBox[] cone_bboxes             # Bounding boxes for cones
#BBox[] disabled_parking_bboxes # Bounding boxes for disabled parking signs
#BBox[] pedestrian_sign_bboxes  # Bounding boxes for pedestrian signs
#BBox[] stop_sign_bboxes        # Bounding boxes for stop signs
#BBox[] traffic_light_bboxes    # Bounding boxes for traffic lights
#float32 obstacle_distance      # 100 if no obstacle
#############################################################################
# BBOX DEFINITIONS
#############################################################################
Centerline[] centerlines        #Endpoints of the center lines in real world
bool are_cones                  #T/F of whether or not there are parking cones
bool is_disabled_parking        #T/F of whether or not there is a disabled parking sign 
BBox disabled_parking_bbox      #Disabled parking bounding box
bool is_pedestrian_sign         #T/F of whether or not there is a pedestrian sign
bool is_pedestrian              #T/F of whether or not there is a pedestrian in the road
bool is_stop_sign               #T/F of whether or not there is a stop sign
bool is_traffic_light           #T/F of whether or not there is a traffic light
bool is_green                   #T/F of whether or not traffic light is green
float32 obstacle_distance       #Distance to an obstacle
float32 TAcar_distance          #Distance to the TA car
float32[] TAcar_loc				#xy coord of bottom midpoint of TA car bbox
float32 pedestrian_distance     #Distance to pedestrian
# cone in real world coords
float32 mid_cone_x
float32 mid_cone_y

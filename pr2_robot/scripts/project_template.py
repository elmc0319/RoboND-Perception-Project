#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import pcl


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    ############### TODO: Convert ROS msg to PCL data ########################
    cloud = ros_to_pcl(pcl_msg)

    ############### TODO: Statistical Outlier Filtering #####################
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)
    # Any point with a mean distance larger than global will be considered out
    outlier_filter.set_std_dev_mul_thresh(0.1)
    cloud_filtered = outlier_filter.filter()
    #pcl.save(cloud_filtered, './pipeline_1_outlier_removal_filter.pcd')

    ############### TODO: Voxel Grid Downsampling ############################
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size
    # Experiment and find the appropriate size!
    LEAF_SIZE = 0.01

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    #pcl.save(cloud_filtered, './pipeline_2_voxel_grid_filter.pcd')

    # TODO: PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.5 #.765
    axis_max = 1.5 #1.3
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_passthrough = passthrough.filter()
    passthrough = cloud_passthrough.make_passthrough_filter()
    #pcl.save(cloud_filtered, './pipeline_3_passthrough_filter.pcd')

    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough.filter()
    #pcl.save(cloud_filtered, './pipeline_pr2_voxel_grid_filter.pcd')

    ################## TODO: RANSAC Plane Segmentation #########################

    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    ############### TODO: Extract inliers and outliers ##############################
    # Extract inliers

    cloud_table = cloud_filtered.extract(inliers, negative=False)
    #pcl.save(cloud_table, './pipeline_pr2_extracted_inliers.pcd')

    # Extract outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
    #pcl.save(cloud_objects, './pipeline_pr2_extracted_outliers.pcd')

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(20)
    ec.set_MaxClusterSize(1500)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Assign a random color to each isolated object in the scene
    cluster_color = get_color_list(len(cluster_indices))

    # Store the detected objects and labels in these lists
    detected_objects_labels = []
    detected_objects = []
    color_cluster_point_list = []

    # Iterate through each detected object cluster for object recognition
    for index, pts_list in enumerate(cluster_indices):

        # Store the object's cloud in this list
        object_cluster = []

        # Create an individual cluster just for the object being processed
        for i, pts in enumerate(pts_list):
            # Retrieve cloud values for the x, y, z, rgb object
            object_cluster.append([cloud_objects[pts][0],
                                   cloud_objects[pts][1],
                                   cloud_objects[pts][2],
                                   cloud_objects[pts][3]])

            # Retrieve cloud values for the x, y, z object, assigning a
            #   preidentified color to all cloud values
            color_cluster_point_list.append([white_cloud[pts][0],
                                             white_cloud[pts][1],
                                             white_cloud[pts][2],
                                             rgb_to_float(cluster_color[index])])


        # Convert list of point cloud features (x,y,z,rgb) into a point cloud
        pcl_cluster = pcl.PointCloud_PointXYZRGB()
        pcl_cluster.from_list(object_cluster)

        # Convert the cluster from pcl to ROS using helper function
        ros_cloud = pcl_to_ros(pcl_cluster)
        #pcl_objects_pub.publish(ros_cloud)

        # Extract histogram features (similar to capture_features.py)
        histogram_bins = 128
        chists = compute_color_histograms(ros_cloud,
                                          nbins=histogram_bins,
                                          using_hsv=True)
        normals = get_normals(ros_cloud)
        nhists = compute_normal_histograms(normals,
                                           nbins=histogram_bins)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result and add it
        #   to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cloud
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Create new cloud containing all clusters, each with a unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_object_cluster = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)

    # Publish ROS messages of the point clouds and detected objects
    pcl_objects_cloud_pub.publish(ros_cloud_object_cluster) # solid color objects
    pcl_objects_pub.publish(ros_cloud_objects)      # original color objects
    pcl_table_pub.publish(ros_cloud_table)          # table cloud
    detected_objects_pub.publish(detected_objects)  # detected object labels

    try:
        # Add some logic to determine whether or not your object detections
        #   are robust enough before calling pr2_mover()
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass



# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    #centroid list:
    centroids = []
    #label list: dict_list
    output_list = []
    #dictionary list (yaml files dict)
    dict_list = []

    # TODO: Get/Read parameters (from output yaml notes)

    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')


    #Define number of objects:
    num_of_objects = len(object_list_param)

    # TODO: Parse parameters into individual variables
    dropbox_name = {}
    dropbox_position = {}
    label_dict = []

    #match list length to parameter list length
    if not len(object_list) == len(object_list_param):
	rospy.loginfo("Object list does not match object pick list.")
	return

    # TODO: Rotate PR2 in place to capture side tables for the collision map
    #Define left (red drop box) and right (teal dropbox) positions from parameter list
    red_dropbox_position = dropbox[0]['position']
    teal_dropbox_position = dropbox[1]['position']

    #Create Q (quality)/accuracy to quantify accuracy of predicted position:
    Q_Score=0

    #Create ground truth list:

    ground_truth = [element['name'] for element in objects]

    #Compare ground truth with detected objects:
    for object in object_list:
    # Initialize prediction
        prediction_label = object_list.label

    # compare prediction with ground truth
        if prediction_label in ground_truth:

            # remove detected label from ground truth
            ground_truth.remove(predicttion_label)

            # count successful prediction
            Q_Score += 1
        else:

            # mark unsuccessful detection
            object_list.label = 'error'

	rospy.loginfo('Detected {} objects out of {}.'.format(Q_Score, num_of_objects))
    # TODO: Loop through the pick list
    sorted_objects = []
    # Convert the numpy float64 to native python floats
    for i in range(num_of_objects):
		pick_list_label = objects[i]['name']

		for detected_object in detected_objects:
			if detected_object.lablel == pick_list_label:
				sorted_objects.append(detected_object)

                # Remove current object
                detected_objects.remove(detected_object)
                break

    # Create lists for dropbox groups

    dropbox_groups = []
    for sorted_object in sorted_objects:

        # Calculate the centroid
        center_points = ros_to_pcl(sorted_object.cloud).to_array()
        centroid = np.mean(center_points, axis=0)[:3]

        # Append centroid as <numpy.float64> data type
        centroids.append(centroid)

        # Search for the matching dropbox group
        # Check if the object to be moved was found in the perception analysis,
        #   populating the pick_pose message if it was
        for object_i, object_val in enumerate(object_list):
            if object_name != object_val.label:
                # Skip if it doesn't match the object
                continue
        
        for pl_item in objects:

            # Compare objects to labdls
            if pl_item['name'] == sorted_object.label:

                # If match found, add group to list:
                dropbox_groups.append(pl_item['group'])
                break

        # TODO: Get the PointCloud for a given object and obtain it's centroid
	for j in range(len(sorted_objects)):

	    object_name = String()
            #object_name = object_list_param[i]['name']
            object_name.data = obj['name']
            #object_group = object_list_param[i]['group']
	    object_group = dropbox_groups[j]

	    #convert to 64 float data type:
	    centroids.append(centroid)
	    np_centroid = centroids[j]
            scalar_centroid = [np.asscalar(element) for element in np_centroid]

        # TODO: Assign the arm to be used for pick_place (copy code from capture_features.py)

	    arm_name = String()
	    if obj['group'] == 'red':
                arm_name.data = 'left'
            elif obj['group'] == 'green':
                arm_name.data = 'right'
            else:
                print "ERROR, group must be Red or Green!"

	    #initialize test scene:
	    test_scene_num = Int32()
	    num_scene =2 #2,3 (world#)
	    test_scene_num.data = num_scene


	    #Create Pick Pose:
	    pick_pose = Pose()
            pick_pose.position.x = scalar_centroid[0]
            pick_pose.position.y = scalar_centroid[1]
            pick_pose.position.z = scalar_centroid[2]

	    # TODO: Create 'place_pose' for the object
	    #Assign pose positions to arm name data matrix:

	    #pose()
	    place_pose.position.x = dict_dropbox[arm_name.data][0]
            place_pose.position.y = dict_dropbox[arm_name.data][1]
            place_pose.position.z = dict_dropbox[arm_name.data][2]

	    # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
	    yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
            dict_list.append(yaml_dict)

            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')

           # try:
                #pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
               # response = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
               # print("Response: ", response.success)

            #except rospy.ServiceException, e
                #print("Service call failed: %s"%e)
            
            # Remove the object from object_list to indicate it was picked up
            del object_list[object_i]
            
            # Stop looking through the other identified objects
            break

    # TODO: Output your request parameters into output yaml file

    yaml_filename = "output_" + str(test_scene_num.data) + ".yaml"

    send_to_yaml(yaml_filename, dict_list)
    print "yaml file created successfully"

#     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e



if __name__ == '__main__':

   
    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # Create Subscriber to receive the published data coming from the
    #   pcl_callback() function that will be processing the point clouds
    pcl_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2,
                               pcl_callback, queue_size=1)

    # Create Publishers
    object_markers_pub = rospy.Publisher('/object_markers', Marker,
                                         queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects',
                                           DetectedObjectsArray,
                                           queue_size=1)
    # Isolated object point cloud with the object's original colors
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    # Isolated object point cloud with random colors
    pcl_objects_cloud_pub = rospy.Publisher('/pcl_objects_cloud', PointCloud2,
                                            queue_size=1)
    # Table point cloud without the objects
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)

    # Load model from disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()


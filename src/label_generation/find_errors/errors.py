"""
Creates the position_angle_error_details folder
"""

import glob
import json
import math
from itertools import permutations 
import numpy as np
import re
import os

def get_points_from_dict(points_dict):
    """
    function to convert dict to list

    Parameters
    ----------
    points_dict: dict
        contains dictionary of (x, y) coordinates

    Returns
    -------
    list
        returns the coordinates in the form of a list
    """
    ret = []
    for i in range(len(points_dict)):
        ret.append((points_dict[i]['x'], points_dict[i]['y']))
    return ret

def find_sum_of_errors(gt_comb, exp_comb):
    """
    Function to find sum of errors. 
    Uses l2 norm to ind errors and then adds them up

    Parameters
    ----------
    gt_comb: numpy array
        contains coordinates
    exp_comb: numpy array
        contains coordinates
    
    Returns
    -------
    int
        sum of l2 norms
    numpy array
        l2 norms between ground truth data and subject data
    """
    # return sum([math.sqrt((gt_comb[i][0] - exp_comb[i][0]) ** 2 + (gt_comb[i][1] - exp_comb[i][1]) ** 2) for i in range(len(gt_comb))])
    l2_norm = np.linalg.norm(gt_comb - exp_comb, axis=1)
    return (sum(l2_norm), l2_norm)

def find_raw_arctan2_angles(first, second):
    """
    Find raw angle in degrees wrt. horizontal axis.

    Parameters
    ----------
    first: numpy array
        contains coordinates
    second: numpy array
    contains coordinates

    Returns
    -------
    float
        contains angles in degrees
    """
    # print(first)
    # print(second)
    diff = second - first
    # print(diff)
    return -np.rad2deg(np.arctan2((diff[1] + 0.0000001), (diff[0] + 0.0000001))) # arctan for finding angle, and rad2deg for converting to degrees.

def find_intersection(a,b,c,d):
    """
    Find intersection point between 2 lines

    Parameters
    ----------
    a, b, c, d: lists
        contains the four coordinates (2 from each line) -  one for the direction of heading of the red aircraft, another for the missed aircraft
    
    Returns
    -------
    floats
        intersection coordinates
    """
    # standard form line eq Line_AB
    a1 = b[1] - a[1]
    b1 = a[0] - b[0]
    c1 = a1*a[0] + b1*a[1]
 
    # standard form line eq Line_CD
    a2 = d[1] - c[1]
    b2 = c[0] - d[0]
    c2 = a2 * c[0] + b2 * c[1]
 
    determinant = a1 * b2 - a2 * b1
 
    if (determinant == 0):
        return math.inf, math.inf
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
    return x, y

def find_intersection_distance(red_gt, intersection_point):
    """
    Find distance from intersection point to red aircraft, and check if the intersection occurs in front of the red aircraft or behind.

    Parameters
    ----------
    red_gt: numpy array
        coordinates for the red aircraft
    
    intersection_point:
        coordinates for the intersection point
    
    Returns
    -------
    float
        distance between red aircraft and the intersection point
    """
    distance = np.linalg.norm(red_gt[0] - intersection_point)
    first_angle = find_raw_arctan2_angles(red_gt[0], red_gt[1])
    second_angle = find_raw_arctan2_angles(red_gt[0], intersection_point)
    if abs(first_angle - second_angle) < 10:                                             # if angles are similar, both points are on the same side, else no
        pass
    else:
        distance = -distance
    return distance

def get_intersection_distance_from_red_gt(red_gt, gt_left_over):
    """
    Wrapper function for intersection distance calculations.

    Parameters
    ----------
    red_gt: numpy array
        coordinates of red aircraft

    gt_leftover: numpy array
        coordinates of the missed aircraft
    
    Returns
    -------
    float
        intersection distance
    """
    intersection_point = find_intersection(red_gt[0], red_gt[1], gt_left_over[0], gt_left_over[1])
    intersection_point = np.array(intersection_point)
    # print(red_gt)
    # print(gt_left_over[i])
    intersection_distance = find_intersection_distance(red_gt, intersection_point)
    return intersection_distance

def get_distance_between_aircrafts(red_gt, gt_left_over):
    """
    Find euclidean distance from red aircraft to missed aircraft

    Parameters
    ----------
    red_gt: numpy array
        coordinates of red aircraft

    gt_leftover: numpy array
        coordinates of the missed aircraft
    
    Returns
    -------
    float
        euclidean distance
    """
    diff = red_gt - gt_left_over
    l2_norm = np.linalg.norm(diff[0])
    return l2_norm

def find_angles(arr):
    """
    Finds the angle of a vector wrt. the horizontal axis and returns it in degrees.
    
    Parameters
    ----------
    arr: numpy array
        has shape (x, 2, 2)
    
    Returns
    -------
    numpy array
        has shape (x,). Contains angles in degrees.
    
    """
    first = arr[:, 0, :]
    second = arr[:, 1, :]
    diff = second - first
    return -np.rad2deg(np.arctan2((diff[:, 1] + 0.0000001), (diff[:, 0] + 0.0000001))) # arctan for finding angle, and rad2deg for converting to degrees.

def find_angle_errors(gt, exp):
    """   
    Finds angle errors by subtraction
    
    Parameters
    ----------
    gt: numpy array
        Contains ground truth data, has shape (x, 2, 2)
    exp: numpy array
        Contains subject data, has shape (x, 2, 2)
    
    Returns
    -------
    numpy array
        has shape (x,). Contains angles of vectors in ground truth
    numpy array
        has shape (x,). Contains angles of vectors in subject data
    numpy array
        has shape (x,). Contains difference in angles (max difference possible is 180).
    """
    gt_angles = find_angles(gt)                                                 # get ground truth angles and subject angles
    exp_angles = find_angles(exp)
    gt_angles = np.where(gt_angles < 0, gt_angles + 360, gt_angles)             # make sure all angles are >= 0
    exp_angles = np.where(exp_angles < 0, exp_angles + 360, exp_angles)
    d = gt_angles - exp_angles                                                  # difference between angles. Can be anywhere from -360 to 360
    d = np.where((np.abs(d) > 180) & ((d) > 0), d - 360, d)                     # handle so that the minimum of the two angles between 2 vectors is returned
    d = np.where((np.abs(d) > 180) & ((d) < 0), d + 360, d)                     # if angle that is clockwise from gt to subject is smaller, d is positive
    return gt_angles, exp_angles, d                                             # else negative. d is between -180 and 180.

def find_best_position_match(gt_points, gt_positions, exp_positions):
    """
    Finds the best matches between ground truth data positions and subject data positions.

    Parameters
    ----------

    gt_points: numpy array
    gt_positions: numpy array
    exp_positions: numpy array

    Returns
    -------
    numpy array
        has shape (x, 2, 2). x indicates the number of aircrafts the subject attempted to predict
    int
        contains sum of l2 norms( the total error in position)
    numpy array
        array of l2 norms
    numpy array
        array containing aircrafts that were missed
    int
        contains the number of aircrafts the subject failed to predict
    """
    exp_num = len(exp_positions)                                              # subject might not have attempted all aircrafts
    gt_num = len(gt_positions)
    best_s = float("inf")                                                     # initialize empty variables
    best_perm = []
    best_perm_set_diff = set()
    gt_left_over = []

    if exp_num == 0:                                                          # handle edge case where subject has not attempted at all
        s, l2_norm = find_sum_of_errors(gt_positions, 0)
        return np.array([]), s, l2_norm, gt_points, len(gt_points)


    l = [i for i in range(1, gt_num + 1)]
    all_permutations = permutations(l, exp_num)                               # get all permutations depending on how many the subject attempted
    for perm in all_permutations:                                             # iterate over all permutations
        set_diff = set(l) - set(perm)                                         # this contains the indices that were not used in the permutation
        perm = np.array(perm, dtype=np.int64) - 1
        gt_perm = gt_positions[perm]                                          # get corresponding postion locations (contains first clicks)
        gt_points_perm = gt_points[perm]                                      # get corresponding points (contains both clicks)
        s, l2_norm = find_sum_of_errors(gt_perm, exp_positions)               # get errors
        if s < best_s:                                                        # if this permutation is better, use this
            best_s = s                                                        # store best parameters
            best_l2 = l2_norm
            best_perm = gt_points_perm
            best_perm_set_diff = set_diff
    
    if best_perm_set_diff:                                                    # handle missing clicks
        best_perm_set_diff = np.array(tuple(best_perm_set_diff), dtype=np.int64) - 1 # contains the indices that weren't used in permutation
        gt_left_over = gt_points[best_perm_set_diff]                       # get corresponding position locations (contains first clicks)
        
    return best_perm, best_s, best_l2, gt_left_over, len(best_perm_set_diff)


def save_errors(gt_path, exp_path, file_name_to_save):
    """
    Wrapper to facilitate core functions and saves the errors in a json file at appropriate locations

    Parameters
    ----------
    gt_path: str
        file location of ground truth data
    exp_path: str
        file location of subject data
    file_name_to_save: str
        file location to store generated data
        
    Returns
    -------
    None

    """

    with open(gt_path, 'r') as f:                                              # open ground truth json file and read
        gt_data = json.load(f)
    
    with open(exp_path, 'r') as f:                                             # open subject json file and read
        exp_data = json.load(f)

    gt_data = gt_data["saved_points"]                                          # we will use only the data in the "saved_points" key
    exp_data = exp_data["saved_points"]

    if not exp_data:
        return
    
    gt_points = get_points_from_dict(gt_data)                                  # call function to convert saved_points dict to list
    exp_points = get_points_from_dict(exp_data)

    if len(exp_points) > len(gt_points):                                       # edge case where subject has predicted more aircraft than was in the trial
        exp_points = exp_points[:len(gt_points)]
    
    if len(exp_points) % 2:                                                    # edge case where odd number of clicks were registered
        exp_points = exp_points[:-1]

    gt_positions = [gt_points[i] for i in range(0, len(gt_points), 2)]         # extract position clicks only (every 2nd click is a click for direction)
    exp_positions = [exp_points[i] for i in range(0, len(exp_points), 2)]

    gt_positions = np.array(gt_positions)                                      # convert to numpy array for easier processing
    exp_positions = np.array(exp_positions)
    gt_points = np.array(gt_points)
    exp_points = np.array(exp_points)                                
    gt_points = gt_points.reshape((-1, 2, 2))
    best_exp = exp_points.reshape((-1, 2, 2))                                  # simply reshape exp_points to suit over needs

    red_gt = gt_points[0]

    best_gt, best_s, best_l2, gt_left_over, num_not_attempted = find_best_position_match(gt_points, gt_positions, exp_positions)    # get best ground truth permutation

    if len(best_gt) == 0 or len(best_exp) == 0:                                # handle edge case where subject has not attempted at all
        angles_not_attempted = np.ones((len(gt_points),), dtype=np.float64) * 180
        angles_abs_sum_not_attempted = np.sum(np.abs(angles_not_attempted))    # error for unattempted trials is maximum regardless of ground truth angles
        dict_to_save = {}                                                      # store all information into dictionary
        dict_to_save["ground_truths"] = gt_points.tolist()
        dict_to_save["red_aircraft"] = red_gt.tolist()
        dict_to_save["best_ground_truth_match"] = best_gt.tolist()
        dict_to_save["subject_guesses"] = best_exp.tolist()
        dict_to_save["l2_norms"] = best_l2.tolist()
        dict_to_save["sum_of_l2"] = best_s
        dict_to_save["gt_angles"] = angles_not_attempted.tolist()
        dict_to_save["subject_angles"] = angles_not_attempted.tolist()
        dict_to_save["angle_errors"] = angles_not_attempted.tolist()
        dict_to_save["abs_sum_of_angles_diff"] = angles_abs_sum_not_attempted

        dict_to_save["number_of_missed_aircrafts"] = num_not_attempted
        dict_to_save["missed_aircrafts"] = []
        dict_to_save["intersection_distances"] = []
        dict_to_save["distances_from_red_aircraft"] = []

    gt_angles, exp_angles, angles_diff = find_angle_errors(best_gt, best_exp)  # get angles errors

    intersection_distances = []
    distances = []

    if len(gt_left_over):
        for i in range(len(gt_left_over)):
            intersection_distance = get_intersection_distance_from_red_gt(red_gt, gt_left_over[i])
            intersection_distances.append(intersection_distance)
            distance =  get_distance_between_aircrafts(red_gt, gt_left_over[i])                                                  # get distances between missed aircraft and red aircraft
            distances.append(distance)
    
    gt_left_over = np.array(gt_left_over)
    intersection_distances = np.array(intersection_distances)
    distances = np.array(distances)

    angles_abs_sum = np.sum(np.abs(angles_diff))                               # total angle error

    dict_to_save = {}                                                          # store all information into dictionary
    dict_to_save["ground_truths"] = gt_points.tolist()
    dict_to_save["red_aircraft"] = red_gt.tolist()
    dict_to_save["best_ground_truth_match"] = best_gt.tolist()
    dict_to_save["subject_guesses"] = best_exp.tolist()
    dict_to_save["l2_norms"] = best_l2.tolist()
    dict_to_save["sum_of_l2"] = best_s
    dict_to_save["gt_angles"] = gt_angles.tolist()
    dict_to_save["subject_angles"] = exp_angles.tolist()
    dict_to_save["angle_errors"] = angles_diff.tolist()
    dict_to_save["abs_sum_of_angles_diff"] = angles_abs_sum

    dict_to_save["number_of_missed_aircrafts"] = num_not_attempted
    dict_to_save["missed_aircrafts"] = gt_left_over.tolist()
    dict_to_save["intersection_distances"] = intersection_distances.tolist()
    dict_to_save["distances_from_red_aircraft"] = distances.tolist()

    with open(file_name_to_save, "w", encoding='utf-8') as outfile: 
        json.dump(dict_to_save, outfile, separators=(',', ':'), indent=4)


if __name__ == "__main__":
    for folder in glob.glob("../../../user_data/*/"):                                   # go through all relevant folders, currently in the COVEE directory
        if re.search("\D*\d{10}", folder):
            folder_name = folder[19:-1]                                         # store folder name for later use
            print(folder)
            for file in glob.glob(folder + "/clicks_data/*.json"):            # go through all relevant files
                if re.search("\D*Task_\d_\d_\d\D*", file):
                    file_name = file[file.rfind("/") + 1:-5]                  # store file name for later use
                    # print(file_name)
                    if re.search("secondary", file_name):                      # to facilitate accessing the correct ground truth file
                        f_name = file_name[:-10]
                    else:
                        f_name = file_name
                    gt_file = "../UAV_ground_truth/" + f_name + ".json"   
                    exp_file = file
                    # file name to save data as json
                    place_to_save = "../position_angle_error_details/" + folder_name + "/"
                    if not os.path.isdir(place_to_save):
                        os.makedirs(place_to_save)   
                    file_name_to_save = place_to_save + "/" + file_name + ".json"
                    print(file_name_to_save)
                    save_errors(gt_file, exp_file, file_name_to_save)      # call wrapper function

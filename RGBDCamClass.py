#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import numpy as np
import cv2
import open3d as o3d
from scipy import spatial
# import open3d_tutorial as o3dtut
from scipy.spatial.transform import Rotation as R
from scipy.optimize import dual_annealing
from numpy.linalg import inv
import socket
import os
import pandas as pd
import collections
import sys
import copy
import time

# sys.path.append('..')
# change to True if you want to interact with the visualization windows
# o3dtut.interactive = not "CI" in os.environ
def VSTARSCam2PCD(path_rgb, path_depth, maxdepth, save=False, path_SaveAs='.\\VSTARSCam2PCD.pcd'):
    # Camera parameters
    F = 17  # Focal length in mm
    H = 18.4  # in mm
    V = 24.6  # in mm
    Fs = 8
    w = int(4096/Fs)  # in pixels
    h = int(3072/Fs)  # in pixels

    px = V / w  # pixel size in mm
    py = H / h  # pixel size in mm

    fx = F / px  # focal length in pixel
    fy = F / py  # focal length in pixel
    cx = w / 2 - 0.5  # principle point x
    cy = h / 2 - 0.5  # principle point y

    # Creating camera object in open3d
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])

    # Import and process depth data

    with open(path_depth, "rb") as fileobj:
        data = np.fromfile(fileobj, dtype='float32')  # dtype='<i4' (little-endian 32bit)

    # Convert 1-dimensional array of format [points...] to 2-dimensional
    depth_arr = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            depth_arr[h - y - 1, x] = data[y * w + x] / np.sqrt(((x - cx) / fx) ** 2 + (
                    (y - cy) / fy) ** 2 + 1)  # correct distance to camera center into depth to plane

    depth_raw = o3d.geometry.Image(np.ascontiguousarray(depth_arr).astype(np.float32))  # create a image object

    # Import and process rgb data
    color_arr = cv2.imread(path_rgb)
    color_raw = o3d.geometry.Image(color_arr)  # creating image object

    # creating rgbd image object, with depth defined
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1,
                                                                    depth_trunc=maxdepth,
                                                                    convert_rgb_to_intensity=False)

    # plt.subplot(1, 2, 1)
    # plt.title('PS RGBD Image')
    # plt.imshow(rgbd_image.color)
    # plt.subplot(1, 2, 2)
    # plt.imshow(rgbd_image.depth)
    # plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam.intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if save == True:
        o3d.io.write_point_cloud(path_SaveAs, pcd, write_ascii=True, compressed=False, print_progress=True)

    return pcd
def preprocess_point_cloud(pcd, voxel_size):
    #     print(":: Downsample with a voxel size %.3f." % voxel_size)
    #     pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    trans_init = np.identity(4);
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result
def clusteringPcd(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=580, min_points=100, print_progress=False))
    max_label = labels.max()

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    #     o3d.visualization.draw_geometries([pcd], zoom=0.8,front=[-1,0, 1], up=[10,-5,100])

    #     print(f"point cloud has {max_label + 1} clusters")

    xyzData = np.asarray(pcd.points)
    pcddf = pd.DataFrame(xyzData, index=labels, columns=['x', 'y', 'z'])

    dqPcd = collections.deque()
    for i in range(0, max_label + 1):
        #         print(i)
        pcddf_np = pcddf.loc[i].to_numpy()

        pcdInit = o3d.geometry.PointCloud()
        pcdInit.points = o3d.utility.Vector3dVector(pcddf_np)
        pcdInit.paint_uniform_color([0.1 * i, 1 - 0.1 * i, 1 - 0.1 * i])
        dqPcd.append(pcdInit)
    return dqPcd, max_label
def overlap(pcd1,pcd2,voxelsize=50):
    comPts=np.empty([1,3])
    KDT_1= spatial.KDTree(pcd1.points)
    outp=KDT_1.query_ball_point(pcd2.points, voxelsize)
    for ind in outp:
        comPts = np.append(comPts,np.asarray(pcd1.points)[ind],axis=0)
    pcdcom = o3d.geometry.PointCloud()
    pcdcom.points = o3d.utility.Vector3dVector(comPts)
    return pcdcom

def frameCoverage(targetcoverage=False):
    # print(pd.Timestamp(time.time(),unit='s'))

    filepath="C:/ProgramData/Tecnomatix/Process Simulate on Teamcenter/17.0/"
    pcd_R = VSTARSCam2PCD(filepath+'OutputFiles/VSTARS_R.png', filepath+'OutputFiles/VSTARS_R.dat', 7500, save=False)
    pcd_L = VSTARSCam2PCD(filepath+'OutputFiles/VSTARS_L.png', filepath+'OutputFiles/VSTARS_L.dat', 7500, save=False)

    # pcd_R = o3d.io.read_point_cloud("./Pictures/rgbd_r.pcd")
    # pcd_L = o3d.io.read_point_cloud("./Pictures/rgbd_l.pcd")

    downpcd_R = pcd_R.voxel_down_sample(voxel_size=50)
    downpcd_L = pcd_L.voxel_down_sample(voxel_size=50)

    # o3d.io.write_point_cloud('./Pictures/rgbd_R_downsample.pcd', downpcd_R, write_ascii=True, compressed=False, print_progress=True)
    # o3d.io.write_point_cloud('./Pictures/rgbd_L_downsample.pcd', downpcd_L, write_ascii=True, compressed=False, print_progress=True)
    # o3d.visualization.draw_geometries([downpcd_R,downpcd_L])
    # o3d.visualization.draw_geometries([pcd_R, pcd_L])
    # downpcd_R = o3d.io.read_point_cloud("./Pictures/rgbd_R_downsample.pcd")
    # downpcd_L = o3d.io.read_point_cloud("./Pictures/rgbd_L_downsample.pcd")

    # In[4]:

    Ref_TM = np.loadtxt('./OutputFiles/RefPos.dat', dtype=float, delimiter=',')
    cam_L_TM = np.loadtxt('./OutputFiles/CamLPos.dat', dtype=float, delimiter=',')
    cam_R_TM = np.loadtxt('./OutputFiles/CamRPos.dat', dtype=float, delimiter=',')
    FrameDat_TM = np.loadtxt('./OutputFiles/FramePos.dat', dtype=float, delimiter=',')

    cam_L_TM_Trans = np.array([cam_L_TM[0,3],cam_L_TM[1,3],cam_L_TM[2,3]])
    cam_R_TM_Trans = np.array([cam_R_TM[0, 3], cam_R_TM[1, 3], cam_R_TM[2, 3]])
    FrameDat_TM_Trans = np.array([FrameDat_TM[0,3],FrameDat_TM[1,3],FrameDat_TM[2,3]])

    uncertainty_L = np.sqrt(sum((FrameDat_TM_Trans[ii] - cam_L_TM_Trans[ii])**2 for ii in range(3)))/100+10
    uncertainty_R = np.sqrt(sum((FrameDat_TM_Trans[ii] - cam_L_TM_Trans[ii]) ** 2 for ii in range(3))) / 100 + 10
    uncertainty = np.max([uncertainty_L, uncertainty_R])


    # In[5]:

    # Camera Frame to Reference Frame Transformation data is required from PS
    # PS captured angles in Euler zyz radian (z1yz2)

    # cam_L_AbsLoc = np.array([7871.47, 4766.32, 1391.3,-2.1,1.58,1.53])
    # cam_R_AbsLoc = np.array([11888.26,5109.1,1150.96,-1.02,1.57,1.57])
    # Ref_AbsLoc = np.array([10303.62,7251.47,500,0,0,np.pi])

    # Change to rotation to z2yz1
    # cam_L_AbsLoc = np.array([7871.47, 4766.32, 1391.3,1.53,1.58,-2.1])
    # cam_R_AbsLoc = np.array([11888.26,5109.1,1150.96,1.57,1.57,-1.02])
    # Ref_AbsLoc = np.array([10303.62,7251.47,500,np.pi,0,0])

    # Camera Lens Rx +90deg
    # Cam_Lens_Rot = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])

    # def Abs2TM(AbsLoc,Rtype):
    #     obj_Rot = R.from_euler(Rtype,AbsLoc[3:6],degrees=False)
    #     obj_TM=np.zeros([4,4])
    #     obj_TM[0:3,0:3]=obj_Rot.as_matrix()
    #     obj_TM[0:3,-1] = np.transpose(AbsLoc[0:3])
    #     obj_TM[-1,-1]=1
    #     return obj_TM

    # cam_L_TM = Abs2TM(cam_L_AbsLoc,'zyz')
    # cam_R_TM = Abs2TM(cam_R_AbsLoc,'zyz')
    # Ref_TM = Abs2TM(Ref_AbsLoc,'zyz')

    camL2RF_TM = np.matmul(inv(cam_L_TM), Ref_TM)
    camR2RF_TM = np.matmul(inv(cam_R_TM), Ref_TM)

    # camL2R_TM = np.matmul(inv(cam_L_TM),cam_R_TM)

    # downpcd_R.paint_uniform_color([1, 0, 0])
    # downpcd_L.paint_uniform_color([0, 0, 1])
    WCS = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=np.array([0., 0., 0.]))
    CAM_L = copy.deepcopy(WCS).transform(inv(camL2RF_TM))
    CAM_R = copy.deepcopy(WCS).transform(inv(camR2RF_TM))

    downpcd_LT = copy.deepcopy(downpcd_L).transform(inv(camL2RF_TM))
    downpcd_RT = copy.deepcopy(downpcd_R).transform(inv(camR2RF_TM))

    # o3d.visualization.draw_geometries([WCS,CAM_L,CAM_R,downpcd_LT,downpcd_RT])

    # In[6]:

    dpcd_LT, dpcd_LT_fpfh = preprocess_point_cloud(downpcd_LT, 50)
    dpcd_RT, dpcd_RT_fpfh = preprocess_point_cloud(downpcd_RT, 50)

    result_icp = refine_registration(dpcd_RT, dpcd_LT, dpcd_RT_fpfh, dpcd_LT_fpfh, 300)

    dpcd_RT.transform(result_icp.transformation)
    # print(result_icp)
    # draw_registration_result(dpcd_R, dpcd_LT, result_icp.transformation)
    o3d.visualization.draw_geometries([WCS,dpcd_RT,dpcd_LT])

    # In[9]:

    leftNumpy = np.asarray(dpcd_LT.points)
    rightNumpy = np.asarray(dpcd_RT.points)
    combineMumpy = np.concatenate((leftNumpy, rightNumpy), axis=0)
    pcdAll = o3d.geometry.PointCloud()
    pcdAll.points = o3d.utility.Vector3dVector(combineMumpy)

    # o3d.visualization.draw_geometries([WCS,pcdAll])

    # In[10]:

    # pcdAll, max_label = clusteringPcd(pcdAll)

    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)
    FrameRF2Dat_TM = inv(FrameDat2RF_TM)
    [x, y, z] = [FrameRF2Dat_TM[0, 3], FrameRF2Dat_TM[1, 3], FrameRF2Dat_TM[2, 3]]
    z_min = z - 2700; z_mx = z_min + 3000;
    y_min = y - 1000; y_mx = y_min + 2000;
    x_min = x - 3500; x_mx = x_min + 4000;

    min_array = np.array([[x_min], [y_min], [z_min]], dtype=float)

    max_array = np.array([[x_mx], [y_mx], [z_mx]], dtype=float)
    crpbxFrame = o3d.cpu.pybind.utility.Vector3dVector([min_array, max_array])
    crpbxFrame = o3d.geometry.AxisAlignedBoundingBox.create_from_points(crpbxFrame)
    pcdFrame = pcdAll.crop(crpbxFrame)
    # o3d.visualization.draw_geometries([WCS, pcdFrame])

    # In[11]:

    # for nn in range(max_label+1):
    #     abdbox=pcdAll[nn].get_axis_aligned_bounding_box()
    #     checker_x = (abdbox.get_min_bound()[0]*abdbox.get_max_bound()[0])<0
    #     checker_y = abdbox.get_max_bound()[1]<0
    #     if checker_x == True and checker_y== True:
    #         FrameInd = nn

    # In[12]:

    # abdbox = pcdAll[FrameInd].get_axis_aligned_bounding_box()
    # abdbox.color = (1, 0, 0)
    # # obdbox = pcdAll[0].get_oriented_bounding_box()
    # # print(obdbox)
    # # obdbox.color = (0, 1, 0)
    # cropbox=[abdbox.get_min_bound(),abdbox.get_max_bound()]
    # # cropbox[0][2]=cropbox[0][2] # AGV height as 415
    # cropbox=o3d.cpu.pybind.utility.Vector3dVector(cropbox)
    # vol_cropbox=o3d.geometry.AxisAlignedBoundingBox.create_from_points(cropbox)
    # pcdFrame=pcdAll[FrameInd].crop(vol_cropbox)
    # o3d.visualization.draw_geometries([WCS,pcdFrame])

    # In[13]:

    meshCrp = o3d.io.read_triangle_mesh("./FrameCAD_Front.ply")
    meshCrp.compute_vertex_normals()

    CAD_dpcd = o3d.io.read_point_cloud("./FrameCAD_Front.pcd")

    # FrameDat_Absloc=np.array([8334,10300.98,2507.66,np.pi/180*0.001,-np.pi/180*0,-1.571])
    # FrameDat_TM = Abs2TM(FrameDat_Absloc,'zyx')
    # print(FrameDat_TM)
    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)

    # DAT=copy.deepcopy(WCS).transform(inv(FrameDat2RF_TM)).paint_uniform_color([1,1,1])

    # FrameDat_WCS=np.array([1490,940,-85,0,-np.pi/2,np.pi/2])
    # FrameWCS2Dat_TM= Abs2TM(FrameDat_WCS,'zyz')

    meshCrp.transform(inv(FrameDat2RF_TM))
    CAD_dpcd.transform(inv(FrameDat2RF_TM))

    CAD_dpcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([WCS,pcdFrame,CAD_dpcd])

    # In[14]:

    pcdframe, pcd_frame_fpfh = preprocess_point_cloud(pcdFrame, 50)
    pcdCAD, pcd_CAD_fpfh = preprocess_point_cloud(CAD_dpcd, 50)

    # voxel_size = 10
    # frame_ransac = execute_global_registration(pcdCAD,pcdframe,pcd_CAD_fpfh, pcd_frame_fpfh,voxel_size)
    # print(frame_ransac)
    # pcdCAD.transform(frame_ransac.transformation)
    # o3d.visualization.draw_geometries([WCS,pcdCAD,pcdframe])

    # In[15]:

    Frame_icp = refine_registration(pcdCAD, pcdframe, pcd_CAD_fpfh, pcd_frame_fpfh, 300)

    pcdCAD.transform(Frame_icp.transformation)
    meshCrp.transform(Frame_icp.transformation)
    Fitness = Frame_icp.fitness
    # print('Frame coverage ratio is',str(Fitness*100),'%')

    # In[16]:

    # o3d.visualization.draw_geometries([pcdCAD,pcdframe])

    # In[17]:

    CADmesh = o3d.t.geometry.TriangleMesh.from_legacy(meshCrp)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(CADmesh)

    query_point = o3d.core.Tensor(np.asarray(pcdframe.points), dtype=o3d.core.Dtype.Float32)

    # Compute distance of the query point from the surface
    unsigned_distance = scene.compute_distance(query_point)
    signed_distance = scene.compute_signed_distance(query_point)
    occupancy = scene.compute_occupancy(query_point)

    reg_pts = np.empty(3)
    for nn in range(len(occupancy)):
        if unsigned_distance.numpy()[nn] < 50:
            reg_pts = np.vstack([reg_pts, pcdframe.points[nn]])

    reg_pts = np.delete(reg_pts, obj=0, axis=0)
    pcdreg = o3d.geometry.PointCloud()
    pcdreg.points = o3d.utility.Vector3dVector(reg_pts)

    # o3d.visualization.draw_geometries([pcdreg,meshCrp])

    # In[18]:

    CADv_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdCAD, voxel_size=50)
    queries = np.asarray(pcdreg.points)
    output = CADv_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    count = 0

    for io in output:
        if io == True:
            count = count + 1

    coverage = count / (len(output))
    # print(coverage*100)

    copyreg = copy.deepcopy(pcdreg).transform(inv(Frame_icp.transformation)).transform(FrameDat2RF_TM)
    now = str(pd.Timestamp(time.time(), unit='s'))
    timestamp = (now[5:10] + now[11:13] + now[14:16])
    Filename = './pcdHistory/reg_pts' + timestamp + '.xyz'
    o3d.io.write_point_cloud(Filename, copyreg, write_ascii=True, compressed=False, print_progress=False)
    # o3d.visualization.draw_geometries([CADv_grid,pcdreg])

    if targetcoverage:
        CAD_dpcd = o3d.io.read_point_cloud("./FrameCAD_Front.pcd")
        pcdTar = o3d.io.read_point_cloud("TargetsMarker.pcd")
        tarcov=targetCoverage(pcdTar, CAD_dpcd, pcdframe)
        print(tarcov)

    return Fitness, coverage, uncertainty
def targetCoverage(pcdtarget, pcdCAD, pcdframe):

    Ref_TM = np.loadtxt('./OutputFiles/RefPos.dat', dtype=float, delimiter=',')
    FrameDat_TM = np.loadtxt('./OutputFiles/FramePos.dat', dtype=float, delimiter=',')
    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)

    pcdframe2 = copy.deepcopy(pcdframe).transform(FrameDat2RF_TM)
    pcdframe2.paint_uniform_color([0, 0 ,1])
    pcdCAD, pcd_CAD_fpfh = preprocess_point_cloud(pcdCAD, 50)
    pcdframe2, pcd_frame2_fpfh = preprocess_point_cloud(pcdframe2, 50)
    Target_icp = refine_registration(pcdframe2, pcdCAD, pcd_frame2_fpfh, pcd_CAD_fpfh, 300)

    pcdframe2.transform(Target_icp.transformation)
    pcdCAD.paint_uniform_color([0.929, 0.576, 0.035])
    # o3d.visualization.draw_geometries([pcdtarget,pcdCAD,pcdframe2])

    # frame_vgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdframe2,voxel_size=30)
    # # o3d.visualization.draw_geometries([pcdtarget,frame_vgrid])
    #
    # queries = np.asarray(pcdtarget.points)
    # output = frame_vgrid.check_if_included(o3d.utility.Vector3dVector(queries))

    count = 0
    KDT= spatial.KDTree(pcdtarget.points)
    outp=KDT.query_ball_point(pcdframe2.points, 23)
    for ind in outp:
        np.asarray(pcdtarget.colors)[ind]=[0,1,0]

    for col in np.asarray(pcdtarget.colors):
        # print(col)
        if col[1] == 1:
            count = count + 1

    # count = 0
    # for ind in range(len(output)):
    #     io = output[ind]
    #     if io == True:
    #         pcdtarget.colors[ind] = [0,1,0]
    #         count = count + 1

    coverage = count / (len(pcdtarget.points)+1)
    # o3d.visualization.draw_geometries([pcdframe2,pcdtarget])
    # o3d.visualization.draw_geometries([pcdtarget])

    return coverage

def frameVisibility(targetcoverage=False):
    filepath = "C:/ProgramData/Tecnomatix/Process Simulate on Teamcenter/17.0/"

    pcd_R = VSTARSCam2PCD(filepath+'/OutputFiles/VSTARS_R.png', filepath+'OutputFiles/VSTARS_R.dat', 6800, save=False)
    pcd_L = VSTARSCam2PCD(filepath+'OutputFiles/VSTARS_L.png', filepath+'OutputFiles/VSTARS_L.dat', 6800, save=False)

    if not targetcoverage:
        vs = 50 # voxel size
    else:
        vs = 20

    downpcd_R = pcd_R.voxel_down_sample(voxel_size=vs)
    downpcd_L = pcd_L.voxel_down_sample(voxel_size=vs)

    # In[4]:

    Ref_TM = np.loadtxt(filepath+'OutputFiles/RefPos.dat', dtype=float, delimiter=',')
    cam_L_TM = np.loadtxt(filepath+'OutputFiles/CamLPos.dat', dtype=float, delimiter=',')
    cam_R_TM = np.loadtxt(filepath+'/OutputFiles/CamRPos.dat', dtype=float, delimiter=',')
    FrameDat_TM = np.loadtxt(filepath+'OutputFiles/FramePos.dat', dtype=float, delimiter=',')

    cam_L_TM_Trans = np.array([cam_L_TM[0,3],cam_L_TM[1,3],cam_L_TM[2,3]])
    cam_R_TM_Trans = np.array([cam_R_TM[0, 3], cam_R_TM[1, 3], cam_R_TM[2, 3]])
    FrameDat_TM_Trans = np.array([FrameDat_TM[0,3],FrameDat_TM[1,3],FrameDat_TM[2,3]])

    uncertainty_L = np.sqrt(sum((FrameDat_TM_Trans[ii] - cam_L_TM_Trans[ii])**2 for ii in range(3)))/100+10
    uncertainty_R = np.sqrt(sum((FrameDat_TM_Trans[ii] - cam_R_TM_Trans[ii]) ** 2 for ii in range(3))) / 100 + 10
    uncertainty = np.max([uncertainty_L, uncertainty_R])


    # In[5]:

    camL2RF_TM = np.matmul(inv(cam_L_TM), Ref_TM)
    camR2RF_TM = np.matmul(inv(cam_R_TM), Ref_TM)

    WCS = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=np.array([0., 0., 0.]))
    CAM_L = copy.deepcopy(WCS).transform(inv(camL2RF_TM))
    CAM_R = copy.deepcopy(WCS).transform(inv(camR2RF_TM))

    downpcd_LT = copy.deepcopy(downpcd_L).transform(inv(camL2RF_TM))
    downpcd_RT = copy.deepcopy(downpcd_R).transform(inv(camR2RF_TM))


    # o3d.visualization.draw_geometries([WCS,CAM_L,CAM_R,downpcd_LT,downpcd_RT])


    # In[6]:

    dpcd_LT, dpcd_LT_fpfh = preprocess_point_cloud(downpcd_LT, 50)
    dpcd_RT, dpcd_RT_fpfh = preprocess_point_cloud(downpcd_RT, 50)

    result_icp = refine_registration(dpcd_RT, dpcd_LT, dpcd_RT_fpfh, dpcd_LT_fpfh, 150)

    dpcd_RT.transform(result_icp.transformation)
    # print(result_icp)
    # draw_registration_result(dpcd_R, dpcd_LT, result_icp.transformation)
    dpcd_LT.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([WCS,dpcd_RT,dpcd_LT])

    # In[9]:

    leftNumpy = np.asarray(dpcd_LT.points)
    rightNumpy = np.asarray(dpcd_RT.points)
    combineMumpy = np.concatenate((leftNumpy, rightNumpy), axis=0)
    pcdAll = o3d.geometry.PointCloud()
    pcdAll.points = o3d.utility.Vector3dVector(combineMumpy)

    # o3d.visualization.draw_geometries([WCS,pcdAll])

    # In[10]:

    # pcdAll, max_label = clusteringPcd(pcdAll)

    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)
    FrameRF2Dat_TM = inv(FrameDat2RF_TM)
    [x, y, z] = [FrameRF2Dat_TM[0, 3], FrameRF2Dat_TM[1, 3], FrameRF2Dat_TM[2, 3]]
    z_min = z - 2100;
    z_mx = z_min + 3000;
    y_min = y - 1000;
    y_mx = y_min + 2000;

    if x>0:
        x_min = x - 3500; x_mx = x_min + 4000;
    else:
        x_min = x - 700; x_mx = x_min + 4000;

    min_array = np.array([[x_min], [y_min], [z_min]], dtype=float)

    max_array = np.array([[x_mx], [y_mx], [z_mx]], dtype=float)
    crpbxFrame = o3d.cpu.pybind.utility.Vector3dVector([min_array, max_array])
    crpbxFrame = o3d.geometry.AxisAlignedBoundingBox.create_from_points(crpbxFrame)
    # crpbxFrame.color = [0,1,0]
    pcdFrame = pcdAll.crop(crpbxFrame)
    # o3d.visualization.draw_geometries([WCS, pcdFrame])

    # In[11]:
    # Calculate Overlapping Area
    pcdLFrame = dpcd_LT.crop(crpbxFrame)
    pcdLFrame.paint_uniform_color([1, 0, 0])
    pcdRFrame = dpcd_RT.crop(crpbxFrame)
    pcdRFrame.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([pcdLFrame, pcdRFrame])

    # pcdLFrame_vgird = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdLFrame, voxel_size=30)
    # queries = np.asarray(pcdRFrame.points)
    # overlap = pcdLFrame_vgird.check_if_included(o3d.utility.Vector3dVector(queries))
    # # o3d.visualization.draw_geometries([pcdLFrame_vgird, pcdRFrame])
    #
    # Frame_olp = np.empty([1, 3])
    # for ii in range(len(queries)):
    #     if overlap[ii]:
    #         Frame_olp = np.concatenate((Frame_olp, [queries[ii, :]]), axis=0)
    #
    # pcdFrame_olp = o3d.geometry.PointCloud()
    # pcdFrame_olp.points = o3d.utility.Vector3dVector(Frame_olp)
    pcdFrame_olp =overlap(pcdLFrame,pcdRFrame,int(vs/2+3))
    pcdFrame_olp.paint_uniform_color([0, 0, 1])
    # o3d.visualization.draw_geometries([pcdFrame_olp])

    # In[12]:


    # In[13]:

    meshCrp = o3d.io.read_triangle_mesh(filepath+"Python Code/FrameCAD_Front.ply")
    meshCrp.compute_vertex_normals()
    CAD_dpcd = o3d.io.read_point_cloud(filepath+"Python Code/FrameCAD_Front.pcd")

    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)
    meshCrp.transform(inv(FrameDat2RF_TM))
    CAD_dpcd.transform(inv(FrameDat2RF_TM))

    CAD_dpcd.paint_uniform_color([0.929,0.82,0.176])
    # o3d.visualization.draw_geometries([pcdFrame_olp,CAD_dpcd])

    # In[14]:

    pcdframe, pcd_frame_fpfh = preprocess_point_cloud(pcdFrame_olp, 50)
    pcdCAD, pcd_CAD_fpfh = preprocess_point_cloud(CAD_dpcd, 50)

    # In[15]:

    Frame_icp = refine_registration(pcdCAD, pcdframe, pcd_CAD_fpfh, pcd_frame_fpfh, 300)

    pcdCAD.transform(Frame_icp.transformation)
    meshCrp.transform(Frame_icp.transformation)
    Fitness = Frame_icp.fitness
    # o3d.visualization.draw_geometries([pcdFrame_olp,CAD_dpcd])
    # print('Frame visibility ratio is',str(Fitness*100),'%')

    coverage = 0

    if targetcoverage:
        CAD_dpcd = o3d.io.read_point_cloud(filepath+"FrameCAD_Front.pcd")
        pcdTar = o3d.io.read_point_cloud(filepath+"TargetsMarker.pcd")
        coverage=targetCoverage(pcdTar, CAD_dpcd, pcdframe)

    return Fitness, coverage, uncertainty

def singleCamValidation(side, targetcoverage=True):

    if side == 'R':
        pcd = VSTARSCam2PCD('./OutputFiles/VSTARS_R_vali.png', './OutputFiles/VSTARS_R_vali.dat', 6800, save=False)
    elif side == "L":
        pcd = VSTARSCam2PCD('./OutputFiles/VSTARS_L_vali.png', './OutputFiles/VSTARS_L_vali.dat', 6800, save=False)

    vs = 20

    downpcd = pcd.voxel_down_sample(voxel_size=vs)

    # In[4]:

    Ref_TM = np.loadtxt('./OutputFiles/RefPos.dat', dtype=float, delimiter=',')
    cam_L_TM = np.loadtxt('./OutputFiles/CamLPos.dat', dtype=float, delimiter=',')
    cam_R_TM = np.loadtxt('./OutputFiles/CamRPos.dat', dtype=float, delimiter=',')
    FrameDat_TM = np.loadtxt('./OutputFiles/FramePos.dat', dtype=float, delimiter=',')

    cam_L_TM_Trans = np.array([cam_L_TM[0,3],cam_L_TM[1,3],cam_L_TM[2,3]])
    cam_R_TM_Trans = np.array([cam_R_TM[0, 3], cam_R_TM[1, 3], cam_R_TM[2, 3]])
    FrameDat_TM_Trans = np.array([FrameDat_TM[0,3],FrameDat_TM[1,3],FrameDat_TM[2,3]])

    uncertainty_L = np.sqrt(sum((FrameDat_TM_Trans[ii] - cam_L_TM_Trans[ii])**2 for ii in range(3)))/100+10
    uncertainty_R = np.sqrt(sum((FrameDat_TM_Trans[ii] - cam_R_TM_Trans[ii]) ** 2 for ii in range(3))) / 100 + 10
    uncertainty = np.max([uncertainty_L, uncertainty_R])


    # In[5]:

    camL2RF_TM = np.matmul(inv(cam_L_TM), Ref_TM)
    camR2RF_TM = np.matmul(inv(cam_R_TM), Ref_TM)

    WCS = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=np.array([0., 0., 0.]))
    CAM_L = copy.deepcopy(WCS).transform(inv(camL2RF_TM))
    CAM_R = copy.deepcopy(WCS).transform(inv(camR2RF_TM))

    if side == 'R':
        cam2RF_TM = camR2RF_TM
    elif side == 'L':
        cam2RF_TM = camL2RF_TM

    downpcd= copy.deepcopy(downpcd).transform(inv(cam2RF_TM))

    o3d.visualization.draw_geometries([WCS,CAM_L,CAM_R,downpcd])


    # In[6]:

    # dpcd, dpcd_fpfh = preprocess_point_cloud(downpcd_LT, 50)
    # dpcd.paint_uniform_color([1,0,0])

    # In[10]:

    # pcdAll, max_label = clusteringPcd(pcdAll)

    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)
    FrameRF2Dat_TM = inv(FrameDat2RF_TM)
    [x, y, z] = [FrameRF2Dat_TM[0, 3], FrameRF2Dat_TM[1, 3], FrameRF2Dat_TM[2, 3]]
    z_min = z - 2100;
    z_mx = z_min + 3000;
    y_min = y - 1000;
    y_mx = y_min + 2000;

    if x>0:
        x_min = x - 3500; x_mx = x_min + 4000;
    else:
        x_min = x - 700; x_mx = x_min + 4000;

    min_array = np.array([[x_min], [y_min], [z_min]], dtype=float)

    max_array = np.array([[x_mx], [y_mx], [z_mx]], dtype=float)
    crpbxFrame = o3d.cpu.pybind.utility.Vector3dVector([min_array, max_array])
    crpbxFrame = o3d.geometry.AxisAlignedBoundingBox.create_from_points(crpbxFrame)
    # crpbxFrame.color = [0,1,0]
    pcdFrame = downpcd.crop(crpbxFrame)
    o3d.visualization.draw_geometries([WCS, pcdFrame])

    # In[11]:

    # In[12]:


    # In[13]:

    meshCrp = o3d.io.read_triangle_mesh("./FrameCAD_Front.ply")
    meshCrp.compute_vertex_normals()
    CAD_dpcd = o3d.io.read_point_cloud("./FrameCAD_Front.pcd")

    FrameDat2RF_TM = np.matmul(inv(FrameDat_TM), Ref_TM)
    meshCrp.transform(inv(FrameDat2RF_TM))
    CAD_dpcd.transform(inv(FrameDat2RF_TM))

    CAD_dpcd.paint_uniform_color([0.929,0.82,0.176])

    o3d.visualization.draw_geometries([pcdFrame,CAD_dpcd])

    # In[14]:

    pcdframe, pcd_frame_fpfh = preprocess_point_cloud(pcdFrame, 50)
    pcdCAD, pcd_CAD_fpfh = preprocess_point_cloud(CAD_dpcd, 50)

    # In[15]:

    Frame_icp = refine_registration(pcdCAD, pcdframe, pcd_CAD_fpfh, pcd_frame_fpfh, 150)

    pcdCAD.transform(Frame_icp.transformation)
    meshCrp.transform(Frame_icp.transformation)
    Fitness = Frame_icp.fitness
    pcdFrame.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcdFrame])
    # print('Frame visibility ratio is',str(Fitness*100),'%')

    coverage = 0
    if targetcoverage:
        CAD_dpcd = o3d.io.read_point_cloud("./FrameCAD_Front.pcd")
        pcdTar = o3d.io.read_point_cloud("TargetsMarker.pcd")
        coverage=targetCoverage(pcdTar, CAD_dpcd, pcdframe)

    return Fitness, coverage, uncertainty


# t0=time.time()
# [fitness,cov,_]=frameVisibility()
# t=time.time()-t0
# print('Frame visibility ratio is',str(fitness*100)[:5],'%')
# print('Target coverage ratio is',str(cov*100)[:5],'%')
# print('Time taken: ' + str(t)+ ' secs')

# t0=time.time()
# _,cov,_ = frameVisibility(True)
# t=time.time()-t0
# print('Target coverage ratio is',str(cov*100)[:5],'%')
# print('Time taken: ' + str(t) + ' secs')

# t0=time.time()
# _,cov,_ = singleCamValidation('L', False)
# t=time.time()-t0
# print('Target coverage ratio is',str(cov*100)[:5],'%')
# print('Time taken: ' + str(t) + ' secs')